//! Commitment cross-session retrieval — live/replay test with LLM judge.
//!
//! This test exercises whether the agent correctly uses the workspace
//! (via `memory_tree`/`memory_read`/`memory_search`) to retrieve
//! previously-captured commitments, rather than relying only on in-context
//! conversation recall. A secondary LLM acts as a semantic judge to
//! verify that the retrieval response actually references both
//! commitments captured earlier in the conversation.
//!
//! # Current scope and a known limitation
//!
//! The test runs as a single multi-turn session. A later turn issues a
//! retrieval query and the test asserts that the agent touched the
//! workspace via a `memory_*` tool. This proves workspace-backed
//! retrieval is working, but does not fully isolate "fresh conversation
//! state, same workspace" — the agent could in principle answer from
//! prior-turn recall alone. Achieving true cross-session isolation
//! (new thread / cleared conversation, preserved workspace) would
//! require adding a `new_thread()` helper to `TestRig`; that is
//! intentionally out of scope here and tracked as a follow-up.
//!
//! # Running
//!
//! Replay mode (default, deterministic, needs a committed trace fixture):
//! ```bash
//! cargo test --features libsql --test e2e_live_commitment_sessions -- --ignored
//! ```
//!
//! Live mode (real LLM calls, records/updates trace fixture):
//! ```bash
//! IRONCLAW_LIVE_TEST=1 cargo test --features libsql \
//!     --test e2e_live_commitment_sessions -- --ignored --test-threads=1
//! ```
//!
//! Live mode requires `~/.ironclaw/.env` with valid LLM credentials
//! (e.g. `ANTHROPIC_API_KEY` for the anthropic backend). The judge
//! runs against the "cheap" LLM configured by `build_provider_chain`.

#[cfg(feature = "libsql")]
mod support;

#[cfg(feature = "libsql")]
mod commitment_session_tests {
    use std::path::PathBuf;
    use std::time::Duration;

    use crate::support::live_harness::{LiveTestHarness, LiveTestHarnessBuilder};

    fn skills_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("skills")
    }

    async fn build_harness(test_name: &str) -> LiveTestHarness {
        LiveTestHarnessBuilder::new(test_name)
            .with_engine_v2(true)
            .with_auto_approve_tools(true)
            .with_max_tool_iterations(40)
            .with_skills_dir(skills_dir())
            .build()
            .await
    }

    /// Send a message and collect only the responses it produced, following
    /// the same pattern as `e2e_live_personas.rs::run_turn`.
    async fn run_turn(
        harness: &LiveTestHarness,
        message: &str,
        expected_responses: usize,
    ) -> Vec<String> {
        let rig = harness.rig();
        let before = rig.wait_for_responses(0, Duration::ZERO).await.len();
        rig.send_message(message).await;
        let responses = rig
            .wait_for_responses(before + expected_responses, Duration::from_secs(300))
            .await;
        let new_responses: Vec<String> = responses
            .into_iter()
            .skip(before)
            .map(|r| r.content)
            .collect();
        assert!(
            !new_responses.is_empty(),
            "Expected at least one response to: {message}"
        );
        new_responses
    }

    const RETRIEVAL_JUDGE_CRITERIA: &str = "\
        The response must reference BOTH commitments that were captured earlier \
        in this conversation:\n\
        (1) Sarah delivering the Q2 budget proposal by Friday, and\n\
        (2) Bob drafting the acquisition term sheet by Tuesday.\n\
        Both owner names (Sarah, Bob), both deliverables (budget proposal, \
        acquisition term sheet), and both timeframes (Friday, Tuesday) should \
        be recognisable. A response that mentions only one commitment, or is \
        vague about who owes what and by when, should FAIL.";

    /// Multi-turn commitment flow: setup → capture two commitments →
    /// retrieval query. Asserts:
    /// 1. During retrieval, the agent invokes a `memory_*` tool (proving
    ///    workspace lookup, not pure conversation recall).
    /// 2. In live mode only, an LLM judge confirms that the retrieval
    ///    response semantically references both captured commitments.
    #[tokio::test]
    #[ignore] // Live tier: requires LLM API keys or a recorded trace fixture
    async fn commitment_cross_session_retrieval() {
        let harness = build_harness("commitment_cross_session_retrieval").await;
        let mut transcript: Vec<(String, Vec<String>)> = Vec::new();

        // ── Session 1: setup + capture ───────────────────────────────────
        let setup_turns = [
            "I'm a CEO. Help me manage my day and track what my team is \
             delivering. Use sensible defaults and skip the configuration \
             questions.",
            "Track this commitment: Sarah is delivering the Q2 budget \
             proposal by Friday.",
            "Track this separately: Bob is drafting the acquisition term \
             sheet by Tuesday next week.",
        ];
        for msg in setup_turns {
            let responses = run_turn(&harness, msg, 1).await;
            transcript.push((msg.to_string(), responses));
        }

        // Record the tool-call boundary before the retrieval turn so we
        // can distinguish retrieval-triggered memory_* calls from any
        // setup-triggered ones (the setup skill also writes to memory).
        let tools_before_retrieval = harness.rig().tool_calls_started();

        // ── Retrieval turn ───────────────────────────────────────────────
        // Explicitly cues workspace lookup via "what's open" which maps
        // onto the commitment-digest skill's activation patterns
        // (keywords: "open commitments", patterns: "what('s| is| are) (pending|open)").
        let retrieval_msg = "What are my open commitments and the main things on my plate today?";
        let retrieval_responses = run_turn(&harness, retrieval_msg, 1).await;

        // Behavioural assertion: the retrieval turn must have touched the
        // workspace. If the agent answered purely from conversation recall,
        // we'd see no new memory_* tool calls here.
        let tools_after_retrieval = harness.rig().tool_calls_started();
        let new_tools: Vec<&String> = tools_after_retrieval
            .iter()
            .skip(tools_before_retrieval.len())
            .collect();
        // `tool_calls_started()` returns formatted names like
        // "memory_tree(commitments/open/)", so prefix-match on the tool name.
        let used_workspace_tool = new_tools.iter().any(|t| {
            let name = t.as_str();
            name.starts_with("memory_tree")
                || name.starts_with("memory_read")
                || name.starts_with("memory_search")
        });
        assert!(
            used_workspace_tool,
            "Expected retrieval turn to call memory_tree/memory_read/memory_search \
             to look up commitments from the workspace. New tools observed: \
             {new_tools:?}. Full tool activity: {tools_after_retrieval:?}"
        );

        // Semantic assertion via LLM judge. Returns `None` in replay mode
        // (no judge provider), so replay relies on the behavioural check
        // above plus the substring sanity check below.
        if let Some(verdict) = harness
            .judge(&retrieval_responses, RETRIEVAL_JUDGE_CRITERIA)
            .await
        {
            assert!(
                verdict.pass,
                "LLM judge rejected retrieval response.\n\
                 Reasoning: {}\n\
                 Response:\n{}",
                verdict.reasoning,
                retrieval_responses.join("\n"),
            );
        }

        // Substring sanity check — cheap, works in both live and replay,
        // and catches regressions where the agent responds without either
        // commitment at all (independent of the judge's verdict quality).
        let joined = retrieval_responses.join("\n").to_lowercase();
        let mentions_sarah_or_budget = joined.contains("sarah") || joined.contains("budget");
        let mentions_bob_or_termsheet = joined.contains("bob") || joined.contains("term sheet");
        assert!(
            mentions_sarah_or_budget && mentions_bob_or_termsheet,
            "Retrieval response should reference both captured commitments. \
             Sarah/budget found: {mentions_sarah_or_budget}, \
             Bob/term-sheet found: {mentions_bob_or_termsheet}. \
             Response:\n{}",
            retrieval_responses.join("\n"),
        );

        transcript.push((retrieval_msg.to_string(), retrieval_responses));
        harness.finish_turns_simple(&transcript).await;
    }
}
