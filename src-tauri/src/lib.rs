mod db;
mod llm;
mod media;
mod models;
mod transcription;

use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use tauri::Manager;

use db::Database;
use models::{
    Candidate, EnvironmentStatus, MediaProbe, NormalizedTranscript, Project, ProjectDetail,
    Transcript, TranscriptWord,
};

#[derive(Clone)]
struct AppState {
    db: Database,
    data_dir: PathBuf,
}

#[tauri::command]
fn environment_status(state: tauri::State<'_, AppState>) -> EnvironmentStatus {
    let llm_provider = std::env::var("LLM_PROVIDER")
        .unwrap_or_else(|_| "deepseek".to_string())
        .to_lowercase();
    EnvironmentStatus {
        data_dir: state.data_dir.to_string_lossy().to_string(),
        has_ffmpeg: media::command_exists("ffmpeg"),
        has_ffprobe: media::command_exists("ffprobe"),
        has_deepgram_key: std::env::var("DEEPGRAM_API_KEY").is_ok(),
        has_anthropic_key: std::env::var("ANTHROPIC_API_KEY").is_ok(),
        has_deepseek_key: std::env::var("DEEPSEEK_API_KEY").is_ok(),
        llm_provider,
    }
}

#[tauri::command]
fn create_project_from_path(
    state: tauri::State<'_, AppState>,
    path: String,
    transcription_mode: String,
) -> Result<Project, String> {
    validate_media_extension(&path).map_err(to_command_error)?;
    let probe = media::probe_media(&path).ok();

    state
        .db
        .create_project(
            &path,
            &transcription_mode,
            probe.and_then(|probe| probe.duration_sec),
        )
        .map_err(to_command_error)
}

#[tauri::command]
fn list_projects(state: tauri::State<'_, AppState>) -> Result<Vec<Project>, String> {
    state.db.list_projects().map_err(to_command_error)
}

#[tauri::command]
fn get_project_detail(
    state: tauri::State<'_, AppState>,
    project_id: String,
) -> Result<ProjectDetail, String> {
    state
        .db
        .project_detail(&project_id)
        .map_err(to_command_error)
}

#[tauri::command]
fn probe_project(
    state: tauri::State<'_, AppState>,
    project_id: String,
) -> Result<MediaProbe, String> {
    let project = state
        .db
        .get_project(&project_id)
        .map_err(to_command_error)?;
    let probe = media::probe_media(&project.source_path).map_err(to_command_error)?;
    state
        .db
        .update_project_status(&project_id, "ingest", probe.duration_sec)
        .map_err(to_command_error)?;
    Ok(probe)
}

#[tauri::command]
fn extract_project_audio(
    state: tauri::State<'_, AppState>,
    project_id: String,
) -> Result<String, String> {
    let project = state
        .db
        .get_project(&project_id)
        .map_err(to_command_error)?;
    let audio_path = media::extract_audio(&project.source_path, &project_dir(&state, &project_id))
        .map_err(to_command_error)?;
    Ok(audio_path.to_string_lossy().to_string())
}

#[tauri::command]
async fn transcribe_project(
    state: tauri::State<'_, AppState>,
    project_id: String,
    provider: String,
    api_key: Option<String>,
) -> Result<Transcript, String> {
    let db = state.db.clone();
    let data_dir = state.data_dir.clone();
    let project = db.get_project(&project_id).map_err(to_command_error)?;
    db.update_project_status(&project_id, "transcribing", None)
        .map_err(to_command_error)?;

    let transcript = match provider.as_str() {
        "deepgram" => {
            let key = api_key
                .or_else(|| std::env::var("DEEPGRAM_API_KEY").ok())
                .ok_or_else(|| {
                    "Set DEEPGRAM_API_KEY or paste an API key to use cloud transcription."
                        .to_string()
                })?;
            let audio_path = media::extract_audio(
                &project.source_path,
                &data_dir.join("projects").join(&project_id),
            )
            .map_err(to_command_error)?;
            transcription::transcribe_deepgram(&audio_path.to_string_lossy(), &key)
                .await
                .map_err(to_command_error)?
        }
        "local" => {
            return Err("Local whisper-rs transcription is reserved for the native model integration pass. Use Deepgram for this MVP build.".to_string());
        }
        other => return Err(format!("Unsupported transcription provider: {other}")),
    };

    let raw_json = serde_json::to_string_pretty(&transcript).map_err(to_command_error)?;
    let saved = db
        .save_transcript(
            &project_id,
            &provider,
            &raw_json,
            Some(&transcript.language),
        )
        .map_err(to_command_error)?;
    db.update_project_status(&project_id, "analyzing", Some(transcript.duration))
        .map_err(to_command_error)?;
    Ok(saved)
}

#[tauri::command]
fn save_demo_transcript(
    state: tauri::State<'_, AppState>,
    project_id: String,
) -> Result<Transcript, String> {
    let transcript = demo_transcript();
    let raw_json = serde_json::to_string_pretty(&transcript).map_err(to_command_error)?;
    let saved = state
        .db
        .save_transcript(&project_id, "demo", &raw_json, Some(&transcript.language))
        .map_err(to_command_error)?;
    state
        .db
        .update_project_status(&project_id, "analyzing", Some(transcript.duration))
        .map_err(to_command_error)?;
    Ok(saved)
}

#[tauri::command]
async fn generate_candidates(
    state: tauri::State<'_, AppState>,
    project_id: String,
    api_key: Option<String>,
    _allow_demo: bool,
) -> Result<Vec<Candidate>, String> {
    let db = state.db.clone();
    let transcript = db
        .latest_transcript(&project_id)
        .map_err(to_command_error)?
        .ok_or_else(|| "Transcribe the project before detecting moments.".to_string())?;
    let normalized: NormalizedTranscript =
        serde_json::from_str(&transcript.raw_json).map_err(to_command_error)?;

    let provider = std::env::var("LLM_PROVIDER")
        .unwrap_or_else(|_| "deepseek".to_string())
        .to_lowercase();

    let drafts = match provider.as_str() {
        "claude" => {
            let key = api_key
                .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
                .ok_or_else(|| "Set ANTHROPIC_API_KEY or supply Claude API Key to generate candidates.".to_string())?;
            llm::detect_candidates_with_claude(&normalized, &key)
                .await
                .map_err(to_command_error)?
        }
        _ => {
            let key = api_key
                .or_else(|| std::env::var("DEEPSEEK_API_KEY").ok())
                .ok_or_else(|| "Set DEEPSEEK_API_KEY or supply DeepSeek API Key to generate candidates.".to_string())?;
            llm::detect_candidates_with_deepseek(&normalized, &key)
                .await
                .map_err(to_command_error)?
        }
    };

    if drafts.is_empty() {
        return Err("No viable clip candidates were returned for this transcript.".to_string());
    }

    let candidates = db
        .replace_candidates(&project_id, &drafts)
        .map_err(to_command_error)?;
    db.update_project_status(&project_id, "ready", None)
        .map_err(to_command_error)?;
    Ok(candidates)
}

#[tauri::command]
fn set_selected_clip_count(
    state: tauri::State<'_, AppState>,
    project_id: String,
    count: usize,
) -> Result<Vec<Candidate>, String> {
    state
        .db
        .set_selected_clip_count(&project_id, count.clamp(0, 10))
        .map_err(to_command_error)
}

#[tauri::command]
fn render_flat_clip_for_candidate(
    state: tauri::State<'_, AppState>,
    candidate_id: String,
) -> Result<String, String> {
    let (candidate, project) = state
        .db
        .get_candidate_with_project(&candidate_id)
        .map_err(to_command_error)?;
    state
        .db
        .update_clip_for_candidate(&candidate_id, "cutting", None, None)
        .map_err(to_command_error)?;

    let output_path = documents_project_dir(&project)?
        .join("clips")
        .join(format!("clip-{:02}_flat.mp4", candidate.rank));

    match media::render_flat_clip(
        &project.source_path,
        candidate.start_sec,
        candidate.end_sec,
        &output_path,
    ) {
        Ok(path) => {
            let path_string = path.to_string_lossy().to_string();
            state
                .db
                .update_clip_for_candidate(&candidate_id, "done", Some(&path_string), None)
                .map_err(to_command_error)?;
            Ok(path_string)
        }
        Err(error) => {
            let message = error.to_string();
            state
                .db
                .update_clip_for_candidate(&candidate_id, "error", None, Some(&message))
                .map_err(to_command_error)?;
            Err(message)
        }
    }
}

#[tauri::command]
fn delete_project(state: tauri::State<'_, AppState>, project_id: String) -> Result<(), String> {
    state.db.delete_project(&project_id).map_err(to_command_error)
}

#[tauri::command]
fn rename_project(
    state: tauri::State<'_, AppState>,
    project_id: String,
    name: String,
) -> Result<(), String> {
    state.db.rename_project(&project_id, &name).map_err(to_command_error)
}

pub fn run() {
    let _ = dotenvy::dotenv();

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let data_dir = app
                .path()
                .app_data_dir()
                .context("resolving app data directory")?;
            std::fs::create_dir_all(&data_dir).context("creating app data directory")?;
            let db = Database::open(&data_dir.join("autoshorts.sqlite"))?;
            app.manage(AppState { db, data_dir });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            environment_status,
            create_project_from_path,
            list_projects,
            get_project_detail,
            probe_project,
            extract_project_audio,
            transcribe_project,
            save_demo_transcript,
            generate_candidates,
            set_selected_clip_count,
            render_flat_clip_for_candidate,
            delete_project,
            rename_project
        ])
        .run(tauri::generate_context!())
        .expect("error while running AutoShorts");
}

fn project_dir(state: &AppState, project_id: &str) -> PathBuf {
    state.data_dir.join("projects").join(project_id)
}

fn documents_project_dir(project: &Project) -> Result<PathBuf, String> {
    let documents_dir = dirs::document_dir()
        .ok_or_else(|| "Could not find your Documents folder for clip output.".to_string())?;
    Ok(documents_dir
        .join("AutoShorts")
        .join(project_output_slug(project)))
}

fn project_output_slug(project: &Project) -> String {
    let stem = std::path::Path::new(&project.source_path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or(&project.id);
    let slug = stem
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string();

    if slug.is_empty() {
        project.id.clone()
    } else {
        slug
    }
}

fn validate_media_extension(path: &str) -> Result<()> {
    let extension = std::path::Path::new(path)
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.to_ascii_lowercase())
        .ok_or_else(|| anyhow!("Selected file does not have an extension"))?;

    let allowed = ["mp4", "mov", "mp3", "wav", "m4a"];
    if allowed.contains(&extension.as_str()) {
        Ok(())
    } else {
        Err(anyhow!(
            "Unsupported file type .{extension}. Use mp4, mov, mp3, wav, or m4a."
        ))
    }
}

fn demo_transcript() -> NormalizedTranscript {
    let lines = [
        "The surprising thing about short-form clips is that the best moment is rarely the loudest moment.",
        "It is usually the point where someone finally says the quiet part plainly and the listener can feel the stakes.",
        "That is why the system needs to understand the transcript as a story, not just search for keywords.",
        "A good clip opens with tension, resolves one idea, and ends before the energy leaks away.",
        "If you can rank those moments consistently, the rendering pipeline becomes much easier to trust.",
        "The creator still decides what represents them, but the machine removes the first exhausting pass through hours of footage.",
        "The goal is not to automate taste completely. The goal is to give taste a faster starting point.",
        "Once the strongest moments are visible, captions and platform copy become finishing work instead of discovery work.",
        "That is the workflow AutoShorts is designed around.",
    ];

    let mut words = Vec::new();
    let mut cursor = 0.0;
    for line in lines {
        for token in line.split_whitespace() {
            let clean = token.to_string();
            let end = cursor + 0.32;
            words.push(TranscriptWord {
                text: clean,
                start: cursor,
                end,
                speaker: Some("A".to_string()),
            });
            cursor = end + 0.08;
        }
        cursor += 0.75;
    }

    let segments = transcription::build_segments(&words);

    NormalizedTranscript {
        language: "en".to_string(),
        duration: cursor,
        speakers: vec!["A".to_string()],
        words,
        segments,
    }
}

fn to_command_error(error: impl std::fmt::Display) -> String {
    error.to_string()
}
