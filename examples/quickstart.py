"""AdaRubric Quickstart — evaluate a simple agent trajectory.

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/quickstart.py

    # OpenRouter:
    export OPENROUTER_API_KEY="sk-or-..."
    python examples/quickstart.py --provider openrouter --model openai/gpt-4o
"""

from __future__ import annotations

import argparse
import asyncio

from adarubric import (
    AdaRubricPipeline,
    TaskDescription,
    Trajectory,
    TrajectoryStep,
)
from adarubric.config import AdaRubricConfig, LLMConfig


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider", choices=["openai", "openrouter", "deepseek", "vllm"], default="openai",
    )
    parser.add_argument("--model", help="Provider model ID, e.g. openai/gpt-4o for OpenRouter")
    parser.add_argument("--base-url", help="Override the provider API base URL")
    parser.add_argument(
        "--reasoning", action=argparse.BooleanOptionalAction, default=None,
        help="Enable or disable OpenRouter/DeepSeek reasoning (default: model setting)",
    )
    args = parser.parse_args()
    task = TaskDescription(
        task_id="demo-001",
        instruction=(
            "Use the weather API to check if it will rain in Tokyo tomorrow, "
            "and if so, suggest indoor activities."
        ),
        domain="Personal Assistant",
        expected_tools=["weather_api", "activity_search"],
    )

    trajectory = Trajectory(
        trajectory_id="traj-demo-001",
        task_id="demo-001",
        steps=[
            TrajectoryStep(
                step_id=0,
                thought="I need to check tomorrow's weather in Tokyo first.",
                action="weather_api",
                action_input={"city": "Tokyo", "date": "tomorrow"},
                observation="Tomorrow: 70% chance of rain, high 18°C, low 12°C.",
            ),
            TrajectoryStep(
                step_id=1,
                thought="It's likely to rain. Let me find indoor activities in Tokyo.",
                action="activity_search",
                action_input={"city": "Tokyo", "type": "indoor", "limit": 5},
                observation=(
                    "1. TeamLab Borderless, 2. Tokyo National Museum, "
                    "3. Akihabara arcades, 4. Shibuya Sky observatory, "
                    "5. Cooking class in Tsukiji"
                ),
            ),
            TrajectoryStep(
                step_id=2,
                thought="I have good options. Let me compile a recommendation.",
                action="respond",
                action_input="compile recommendation",
                observation="Response delivered to user.",
            ),
        ],
        final_answer=(
            "It will likely rain in Tokyo tomorrow (70% chance). "
            "Here are 5 indoor activities: TeamLab Borderless, Tokyo National Museum, "
            "Akihabara arcades, Shibuya Sky, and a Tsukiji cooking class."
        ),
    )

    llm = LLMConfig(
        provider=args.provider, base_url=args.base_url, reasoning_enabled=args.reasoning,
    )
    if args.model:
        llm.model = args.model
    config = AdaRubricConfig(llm=llm)
    pipeline = AdaRubricPipeline.from_config(config)

    result = await pipeline.run(task, [trajectory], num_dimensions=5)

    print("=== AdaRubric Evaluation Results ===")
    print(f"Task: {task.instruction[:80]}...")
    print(f"Rubric dimensions: {result.rubric.dimension_names}")
    print(f"Global score: {result.mean_score:.2f}/5.0")
    print(f"Survival rate: {result.survival_rate:.0%}")

    for ev in result.all_evaluations:
        print(f"\n--- Trajectory: {ev.trajectory_id} ---")
        print(f"  Global score: {ev.global_score:.2f}")
        for dim_name, dim_score in ev.dimension_global_scores.items():
            print(f"  {dim_name}: {dim_score:.2f}")
        for step_ev in ev.step_evaluations:
            print(f"  Step {step_ev.step_id}: {step_ev.step_quality_summary}")


if __name__ == "__main__":
    asyncio.run(main())
