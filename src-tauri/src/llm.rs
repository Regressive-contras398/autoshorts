use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::models::{CandidateDraft, NormalizedTranscript, TranscriptSegment};

#[derive(Debug, Deserialize)]
struct CandidateEnvelope {
    candidates: Vec<CandidateDraft>,
}

#[derive(Debug, Deserialize)]
struct AnthropicMessage {
    content: Vec<AnthropicContent>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContent {
    text: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct DeepseekMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct DeepseekChoice {
    message: DeepseekMessage,
}

#[derive(Debug, Deserialize)]
struct DeepseekResponse {
    choices: Vec<DeepseekChoice>,
}

pub async fn detect_candidates_with_deepseek(
    transcript: &NormalizedTranscript,
    api_key: &str,
) -> Result<Vec<CandidateDraft>> {
    let segments = compact_segments(&transcript.segments);
    let prompt = format!(
        "You are identifying the most viral moments and strongest short-form clip candidates from a long-form transcript. \
For each candidate, the clip must be self-contained, starting with an extremely engaging hook within the first 3 seconds (to capture immediate attention on social feeds), \
30-90 seconds long, and cut at clean sentence/thought boundaries. Favor highly shareable content: concrete stories, \
strong opinions, emotional turns, surprising or counter-intuitive claims, clear payoffs, and high-energy/dramatic peaks. \
Avoid rambling setup, context-dependent references, and pure filler. Return up to 10 candidates as JSON matching this schema: \
{{\"candidates\":[{{\"start\":0.0,\"end\":0.0,\"score\":0.0,\"hook\":\"...\",\"rationale\":\"...\"}}]}}\n\nTranscript:\n{segments}"
    );

    let response = reqwest::Client::new()
        .post("https://api.deepseek.com/chat/completions")
        .header("Authorization", format!("Bearer {api_key}"))
        .json(&json!({
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.2,
            "response_format": {
                "type": "json_object"
            }
        }))
        .send()
        .await
        .context("calling DeepSeek")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!("DeepSeek request failed ({status}): {body}"));
    }

    let res_body: DeepseekResponse = response.json().await.context("parsing DeepSeek response")?;
    let text = res_body
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .ok_or_else(|| anyhow!("DeepSeek response did not include choices content"))?;

    parse_candidate_json(&text)
}

#[derive(Debug, Serialize)]
struct ClaudeMessage<'a> {
    role: &'a str,
    content: String,
}

pub async fn detect_candidates_with_claude(
    transcript: &NormalizedTranscript,
    api_key: &str,
) -> Result<Vec<CandidateDraft>> {
    let segments = compact_segments(&transcript.segments);
    let prompt = format!(
        "You are identifying the most viral moments and strongest short-form clip candidates from a long-form transcript. \
For each candidate, the clip must be self-contained, starting with an extremely engaging hook within the first 3 seconds (to capture immediate attention on social feeds), \
30-90 seconds long, and cut at clean sentence/thought boundaries. Favor highly shareable content: concrete stories, \
strong opinions, emotional turns, surprising or counter-intuitive claims, clear payoffs, and high-energy/dramatic peaks. \
Avoid rambling setup, context-dependent references, and pure filler. Return up to 10 candidates as JSON only: \
{{\"candidates\":[{{\"start\":0,\"end\":0,\"score\":0.0,\"hook\":\"...\",\"rationale\":\"...\"}}]}}\n\nTranscript:\n{segments}"
    );

    let model =
        std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-3-5-sonnet-latest".to_string());

    let response = reqwest::Client::new()
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .json(&json!({
            "model": model,
            "max_tokens": 1800,
            "temperature": 0.2,
            "messages": [
                ClaudeMessage {
                    role: "user",
                    content: prompt,
                }
            ]
        }))
        .send()
        .await
        .context("calling Claude")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!("Claude request failed ({status}): {body}"));
    }

    let message: AnthropicMessage = response.json().await.context("parsing Claude response")?;
    let text = message
        .content
        .into_iter()
        .find_map(|content| content.text)
        .ok_or_else(|| anyhow!("Claude response did not include text content"))?;

    parse_candidate_json(&text)
}

pub fn demo_candidates(transcript: &NormalizedTranscript) -> Vec<CandidateDraft> {
    transcript
        .segments
        .chunks(8)
        .take(10)
        .enumerate()
        .filter_map(|(index, chunk)| {
            let first = chunk.first()?;
            let last = chunk.last()?;
            let duration = last.end - first.start;
            if duration < 12.0 {
                return None;
            }

            let end = if duration > 90.0 {
                first.start + 90.0
            } else {
                last.end
            };

            Some(CandidateDraft {
                start: first.start,
                end,
                score: (0.86 - (index as f64 * 0.035)).max(0.55),
                hook: first
                    .text
                    .split_whitespace()
                    .take(12)
                    .collect::<Vec<_>>()
                    .join(" "),
                rationale: "Demo ranking generated without Claude. Use ANTHROPIC_API_KEY for production-quality moment detection.".to_string(),
            })
        })
        .collect()
}

fn compact_segments(segments: &[TranscriptSegment]) -> String {
    segments
        .iter()
        .map(|segment| {
            let speaker = segment.speaker.as_deref().unwrap_or("Speaker");
            format!(
                "[{:.2}-{:.2}] {}: {}",
                segment.start, segment.end, speaker, segment.text
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn parse_candidate_json(text: &str) -> Result<Vec<CandidateDraft>> {
    let trimmed = text
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    let envelope: CandidateEnvelope =
        serde_json::from_str(trimmed).context("parsing candidate JSON")?;

    let mut candidates = envelope
        .candidates
        .into_iter()
        .filter(|candidate| {
            candidate.end > candidate.start
                && candidate.score >= 0.0
                && candidate.score <= 1.0
                && !candidate.hook.trim().is_empty()
        })
        .collect::<Vec<_>>();

    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    candidates.truncate(10);
    Ok(candidates)
}
