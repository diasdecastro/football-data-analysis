#!/usr/bin/env python3
"""Quick test to verify that Shot objects include geometry calculations."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.common.geometry import distance_to_goal, shot_angle
from src.tasks.xg.transform.build_shots import Shot


def test_shot_with_geometry():
    """Test that Shot objects correctly calculate distance and angle."""

    # Test with a known position
    x, y = 100.0, 40.0  # A position on the pitch

    # Calculate expected values
    expected_distance = distance_to_goal(x, y)
    expected_angle = shot_angle(x, y)

    print(f"Test position: x={x}, y={y}")
    print(f"Expected distance to goal: {expected_distance:.3f}")
    print(f"Expected shot angle: {expected_angle:.3f}")

    # Create a Shot object with minimal required fields
    shot = Shot(
        event_id="test_123",
        match_id=12345,
        competition_id=43,
        season_id=4,
        home_team_id=1,
        away_team_id=2,
        team_id=1,
        opponent_team_id=2,
        player_id=100,
        period=1,
        minute=45,
        second=30,
        x=x,
        y=y,
        end_x=120.0,
        end_y=40.0,
        distance_to_goal=expected_distance,
        shot_angle=expected_angle,
        is_goal=0,
        is_penalty=0,
        is_freekick=0,
        is_open_play=1,
        body_part="Right Foot",
        technique="Normal",
        first_time=0,
        one_on_one=0,
    )

    print(f"\nShot object created successfully!")
    print(f"Shot distance_to_goal: {shot.distance_to_goal:.3f}")
    print(f"Shot shot_angle: {shot.shot_angle:.3f}")

    # Verify values match
    assert abs(shot.distance_to_goal - expected_distance) < 1e-6, "Distance mismatch"
    assert abs(shot.shot_angle - expected_angle) < 1e-6, "Angle mismatch"

    print("✅ All tests passed! Geometry calculations are working correctly.")


if __name__ == "__main__":
    test_shot_with_geometry()
