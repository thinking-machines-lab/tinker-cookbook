# Paper's Evaluation Suite

## Benchmarks from the paper (Table 1, 2, 3, 4, 5)

### Math
| Benchmark | Paper Reports | Our Coverage |
|-----------|--------------|-------------|
| IMO 2025 | Yes | NOT BUILT |
| IMO-AnswerBench | Yes | NOT BUILT |
| AIME 2025 | Yes (97.9% with tools) | NOT BUILT |
| AIME 2026 | Yes | NOT BUILT |
| HMMT Feb 2025 | Yes | NOT BUILT |
| MATH-500 | Yes | BUILT ✓ |
| GSM8K | Implied | BUILT ✓ |

### Code
| Benchmark | Paper Reports | Our Coverage |
|-----------|--------------|-------------|
| IOI 2025 | Yes | NOT BUILT |
| ICPC World Finals 2025 | Yes | NOT BUILT |
| LiveCodeBench v6 | Yes | NOT BUILT (data script broken) |
| LiveCodeBench Pro | Yes | NOT BUILT |
| SciCode | Yes | NOT BUILT |
| Codeforces ELO | Yes | NOT BUILT |

### Knowledge & STEM
| Benchmark | Paper Reports | Our Coverage |
|-----------|--------------|-------------|
| MMLU-Redux | Yes | NOT BUILT (similar to MMLU-Pro) |
| MMLU-Pro | Yes | BUILT ✓ |
| GPQA-Diamond | Yes | NOT BUILT |
| HLE (Humanity's Last Exam) | Yes | NOT BUILT |

### Instruction Following & Alignment
| Benchmark | Paper Reports | Our Coverage |
|-----------|--------------|-------------|
| ArenaHard v2 | Yes | NOT BUILT |
| IFBench | Yes | BUILT (as IFEval) ✓ |
| Scale AI Multi-Challenge | Yes | NOT BUILT |

### Long Context
| Benchmark | Paper Reports | Our Coverage |
|-----------|--------------|-------------|
| AA-LCR | Yes | NOT BUILT |
| LongBench v2 | Yes | NOT BUILT |
| NIAH@1M | Yes | NOT BUILT |
| CL-Bench | Yes | NOT BUILT |

### Agentic
| Benchmark | Paper Reports | Our Coverage |
|-----------|--------------|-------------|
| BFCL v4 | Yes | NOT BUILT |
| tau2-Bench | Yes | NOT BUILT |
| Terminal Bench 2.0 | Yes | NOT BUILT |
| SWE-bench Verified | Yes | NOT BUILT |

### Multilingual
| Benchmark | Paper Reports | Our Coverage |
|-----------|--------------|-------------|
| MMLU-ProX | Yes | NOT BUILT |
| WMT24++ | Yes | NOT BUILT |

## Priority for Our Replication

### Tier 1 (most important, feasible)
- GPQA-Diamond — hard science QA, available on HuggingFace
- ArenaHard v2 — alignment/chat quality
- LongBench v2 — long context comprehension

### Tier 2 (important, more work needed)
- AIME 2025 — math reasoning (need dataset)
- LiveCodeBench — code (need working data loader)
- SWE-bench Verified — agentic coding

### Tier 3 (nice to have)
- BFCL v4 — function calling
- MMLU-ProX — multilingual
- Codeforces ELO — competitive programming
