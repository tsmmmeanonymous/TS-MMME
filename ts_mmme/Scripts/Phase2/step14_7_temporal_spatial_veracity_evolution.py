#!/usr/bin/env python3
"""
====================================================================
MME-TSE
STEP 14.7 : TEMPORAL-SPATIAL VERACITY EVOLUTION ANALYZER (TS-VEA)
====================================================================

Purpose
-------
Analyse the retrieved temporal-spatial memories to explain
how historical veracity patterns evolve for an incoming article.

This module DOES NOT modify the final veracity prediction.
Instead, it provides an interpretable evolution analysis
using the retrieved temporal-spatial memory nodes.

Inputs
------
Retrieved Memories from Step 14.4

Each retrieved memory should contain

    memory_index
    topic
    year_month
    location

    retrieval_score

    truth_ratio
    rumor_ratio

    truth_count
    rumor_count

Outputs
-------
Evolution Score
Evolution Confidence
Dominant Historical State
Temporal Trend
Timeline
Transition Summary

====================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import statistics



# ==========================================================
# Configuration
# ==========================================================

TRUTH_THRESHOLD = 0.60
RUMOR_THRESHOLD = 0.60


# ==========================================================
# Data Structure
# ==========================================================

@dataclass
class EvolutionResult:
    """
    Stores the complete evolution analysis.

    This object will later be returned by

        analyse_temporal_spatial_evolution()

    and directly embedded inside the Phase-3
    inference result dictionary.
    """

    evolution_score: float

    evolution_confidence: float

    dominant_state: str

    temporal_trend: str

    retrieved_memory_count: int

    timeline: List[Dict[str, Any]]

    transition_summary: List[str]

    statistics: Dict[str, Any]

    def to_dict(self):

        return asdict(self)


# ==========================================================
# Utility
# ==========================================================

def classify_memory_state(
    truth_ratio: float,
    rumor_ratio: float
):
    """
    Convert Truth/Rumor ratio
    into a semantic state.

    Returns

        Truth
        Rumor
        Mixed
    """

    if truth_ratio >= TRUTH_THRESHOLD:
        return "Truth"

    if rumor_ratio >= RUMOR_THRESHOLD:
        return "Rumor"

    return "Mixed"


# ==========================================================
# Public API
# ==========================================================

# ==========================================================
# Evolution Score
# ==========================================================

def compute_evolution_score(retrieved_memories):
    """
    Weighted Evolution Score.

    E = Σ α_i (truth_ratio_i - rumor_ratio_i)

    where

        α_i = normalized retrieval score
    """

    if len(retrieved_memories) == 0:
        return 0.0

    scores = [
        max(m["retrieval_score"], 1e-8)
        for m in retrieved_memories
    ]

    total = sum(scores)

    weights = [
        s / total
        for s in scores
    ]

    evolution_score = 0.0

    for weight, memory in zip(weights, retrieved_memories):

        truth_ratio = memory["truth_ratio"]
        rumor_ratio = memory["rumor_ratio"]

        local_score = truth_ratio - rumor_ratio

        evolution_score += weight * local_score

    return float(evolution_score)


# ==========================================================
# Evolution Confidence
# ==========================================================

def compute_evolution_confidence(
    retrieved_memories,
    evolution_score
):
    """
    Confidence of the historical trend.

    Higher when

        • retrieval scores agree
        • evolution score is strong

    Output

        0 ~ 1
    """

    if len(retrieved_memories) == 0:
        return 0.0

    retrieval_scores = [
        m["retrieval_score"]
        for m in retrieved_memories
    ]

    avg_similarity = (
        sum(retrieval_scores)
        / len(retrieved_memories)
    )

    confidence = (
        abs(evolution_score)
        + avg_similarity
    ) / 2.0

    confidence = max(
        0.0,
        min(1.0, confidence)
    )

    return float(confidence)


# ==========================================================
# Dominant State
# ==========================================================

def determine_dominant_state(
    evolution_score
):
    """
    Convert evolution score
    into an interpretable state.
    """

    if evolution_score >= 0.30:
        return "Truth"

    if evolution_score <= -0.30:
        return "Rumor"

    return "Mixed"

# ==========================================================
# Timeline
# ==========================================================

def generate_timeline(
    retrieved_memories
):
    """
    Build a chronological timeline
    from retrieved memories.
    """

    timeline = []

    memories = sorted(
        retrieved_memories,
        key=lambda x: x["year_month"]
    )

    for memory in memories:

        state = classify_memory_state(
            memory["truth_ratio"],
            memory["rumor_ratio"]
        )

        timeline.append({

            "month":
                memory["year_month"],

            "location":
                memory["location"],

            "truth_ratio":
                memory["truth_ratio"],

            "rumor_ratio":
                memory["rumor_ratio"],

            "retrieval_score":
                memory["retrieval_score"],

            "state":
                state
        })

    return timeline


# ==========================================================
# Transition Summary
# ==========================================================

def generate_transition_summary(
    timeline
):
    """
    Example

    Rumor

      ↓

    Mixed

      ↓

    Truth
    """

    if len(timeline) == 0:
        return []

    transitions = []

    previous = timeline[0]["state"]

    transitions.append(previous)

    for node in timeline[1:]:

        current = node["state"]

        if current != previous:

            transitions.append(current)

            previous = current

    return transitions


# ==========================================================
# Statistics
# ==========================================================

def generate_statistics(
    retrieved_memories,
    timeline
):

    if not retrieved_memories:
        return {
            "truth_nodes": 0,
            "rumor_nodes": 0,
            "mixed_nodes": 0,
            "average_truth_ratio": 0.0,
            "average_rumor_ratio": 0.0,
            "average_similarity": 0.0,
            "historical_stability": 0.0,
        }

    truth_nodes = sum(
        1
        for t in timeline
        if t["state"] == "Truth"
    )

    rumor_nodes = sum(
        1
        for t in timeline
        if t["state"] == "Rumor"
    )

    mixed_nodes = sum(
        1
        for t in timeline
        if t["state"] == "Mixed"
    )

    avg_truth_ratio = (
        sum(
            m["truth_ratio"]
            for m in retrieved_memories
        )
        / len(retrieved_memories)
    )

    avg_rumor_ratio = (
        sum(
            m["rumor_ratio"]
            for m in retrieved_memories
        )
        / len(retrieved_memories)
    )

    avg_similarity = (
        sum(
            m["retrieval_score"]
            for m in retrieved_memories
        )
        / len(retrieved_memories)
    )

    
    scores = [
        m["truth_ratio"] - m["rumor_ratio"]
        for m in retrieved_memories
    ]

    if len(scores) > 1:
        historical_stability = max(
            0.0,
            1.0 - statistics.pstdev(scores)
        )
    else:
        historical_stability = 1.0

    return {

        "truth_nodes": truth_nodes,

        "rumor_nodes": rumor_nodes,

        "mixed_nodes": mixed_nodes,

        "average_truth_ratio":
            avg_truth_ratio,

        "average_rumor_ratio":
            avg_rumor_ratio,

        "average_similarity":
            avg_similarity,

        "historical_stability": 
            historical_stability
    }


# ==========================================================
# Evolution Trend
# ==========================================================

def determine_temporal_trend(
    transitions
):

    if not transitions:
        return "Unknown"

    if len(transitions) == 1:
        return transitions[0]

    return " → ".join(transitions)


def determine_evolution_direction(transitions):

    if not transitions:
        return "Unknown"

    if len(transitions) == 1:

        if transitions[0] == "Truth":
            return "Stable Truth"

        if transitions[0] == "Rumor":
            return "Stable Rumor"

        return "Stable Mixed"

    first = transitions[0]
    last = transitions[-1]

    if first == "Rumor" and last == "Truth":
        return "Growing Truth"

    if first == "Truth" and last == "Rumor":
        return "Growing Rumor"

    if first == last:
        return f"Stable {first}"

    return "Fluctuating"

# ==========================================================
# Updated Main Function
# ==========================================================

def analyse_temporal_spatial_evolution(
    retrieved_memories
):

    evolution_score = compute_evolution_score(
        retrieved_memories
    )

    evolution_confidence = (
        compute_evolution_confidence(
            retrieved_memories,
            evolution_score
        )
    )

    dominant_state = determine_dominant_state(
        evolution_score
    )

    timeline = generate_timeline(
        retrieved_memories
    )

    transitions = (
        generate_transition_summary(
            timeline
        )
    )

    trend = determine_temporal_trend(
    transitions
    )

    direction = determine_evolution_direction(
        transitions
    )

    statistics = generate_statistics(
        retrieved_memories,
        timeline
    )

    statistics["evolution_direction"] = direction

    return EvolutionResult(

        evolution_score=evolution_score,

        evolution_confidence=
            evolution_confidence,

        dominant_state=dominant_state,

        temporal_trend=trend,

        retrieved_memory_count=len(
            retrieved_memories
        ),

        timeline=timeline,

        transition_summary=transitions,

        statistics=statistics
    )
# ==========================================================
# Export Evolution Results
# ==========================================================

import json
import csv
from pathlib import Path


def save_evolution_json(
    evolution_result,
    output_file
):
    """
    Save complete evolution analysis as JSON.
    """

    output_file = Path(output_file)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            evolution_result.to_dict(),
            f,
            indent=4,
            ensure_ascii=False
        )


# ==========================================================
# Save Timeline
# ==========================================================

def save_timeline_csv(
    timeline,
    output_file
):

    output_file = Path(output_file)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Month",
            "Location",
            "Truth_Ratio",
            "Rumor_Ratio",
            "Evolution_Score",
            "Retrieval_Score",
            "Historical_State"
        ])

        for item in timeline:

            writer.writerow([
                item["month"],
                item["location"],
                item["truth_ratio"],
                item["rumor_ratio"],
                item["truth_ratio"] - item["rumor_ratio"],
                item["retrieval_score"],
                item["state"]
            ])


# ==========================================================
# Console Display
# ==========================================================

def display_evolution_result(
    evolution_result,
    log_fn=print
):

    log_fn("")
    log_fn("=" * 60)
    log_fn("TEMPORAL-SPATIAL VERACITY EVOLUTION")
    log_fn("=" * 60)

    log_fn(
        f"Evolution Score      : "
        f"{evolution_result.evolution_score:.4f}"
    )

    log_fn(
        f"Evolution Confidence : "
        f"{evolution_result.evolution_confidence:.4f}"
    )

    log_fn(
        f"Dominant State       : "
        f"{evolution_result.dominant_state}"
    )

    log_fn(
        f"Evolution Direction : "
        f"{evolution_result.statistics['evolution_direction']}"
    )

    log_fn(
        f"Historical Trend     : "
        f"{evolution_result.temporal_trend}"
    )

    log_fn("")
    log_fn("Timeline")
    log_fn("-" * 60)

    for node in evolution_result.timeline:

        log_fn(
            f"{node['month']} | "
            f"{node['location']} | "
            f"{node['state']} | "
            f"T={node['truth_ratio']:.2f} "
            f"R={node['rumor_ratio']:.2f}"
        )

    log_fn("")
    log_fn("Statistics")
    log_fn("-" * 60)

    for key, value in evolution_result.statistics.items():

        log_fn(f"{key:25s}: {value}")

    log_fn("=" * 60)


# ==========================================================
# Batch Export
# ==========================================================

def export_evolution_analysis(
    evolution_result,
    output_directory
):

    output_directory = Path(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    save_evolution_json(

        evolution_result,

        output_directory /
        "evolution_result.json"

    )

    save_timeline_csv(

        evolution_result.timeline,

        output_directory /
        "evolution_timeline.csv"

    )