# Constrained Decoding Example Models

This directory contains example Triton Python backend models demonstrating various constrained decoding libraries. Each model shows how to integrate a different library to generate structured output from language models.

## Overview

| Model | Library | Approach | Best For |
|-------|---------|----------|----------|
| `outlines-example` | [Outlines](https://github.com/outlines-dev/outlines) | Token masking via JSON Schema/regex/grammar | Production use, arbitrary schemas |
| `guidance-example` | [Guidance](https://github.com/guidance-ai/guidance) | Template DSL with interleaved generation | Complex templates, control flow |
| `lmql-example` | [LMQL](https://github.com/eth-sri/lmql) | SQL-like query language | Declarative constraints |
| `jsonformer-example` | [Jsonformer](https://github.com/1rgs/jsonformer) | Structural token generation | Simple JSON, lightweight |
| `instructor-example` | [Instructor](https://github.com/jxnl/instructor) | Pydantic validation + retry | API-based models, validation |

## Quick Comparison

### Token-Level Masking (Outlines, Guidance, LMQL)
- Constraints are enforced at each generation step
- Invalid tokens are masked (set to -infinity probability)
- **Guaranteed** to produce valid output
- Higher overhead per token

### Structural Generation (Jsonformer)
- Structure tokens (braces, colons) generated deterministically
- LLM only fills in values (strings, numbers)
- Very lightweight, guaranteed structure
- Limited to JSON only

### Validation + Retry (Instructor)
- Generate freely, then validate
- Retry with error feedback if invalid
- Works with any API-based model
- May require multiple attempts

## Installation

Each model requires its specific library. Install them as needed:

```bash
# Outlines
pip install outlines torch transformers

# Guidance
pip install guidance torch transformers

# LMQL
pip install lmql torch transformers

# Jsonformer
pip install jsonformer torch transformers

# Instructor (simulation for local models)
pip install instructor pydantic openai transformers torch
```

## Usage

### Deploying to Triton

To use these models with Triton, copy them to your model repository:

```bash
# Copy a model to the Triton repo
cp -r scripts/models/outlines-example triton-repo/models/

# Reload Triton or restart
curl -X POST "http://localhost:8080/v2/repository/models/outlines-example/load"
```

### Example Requests

#### Outlines - JSON Schema Constrained
```bash
curl -X POST "http://localhost:8080/v2/models/outlines-example/infer" \
  -H "Content-Type: application/json" \
  -d '{"inputs": [
    {"name": "prompt", "shape": [1], "datatype": "BYTES", "data": ["What is the capital of France?"]},
    {"name": "schema_name", "shape": [1], "datatype": "BYTES", "data": ["qa"]}
  ]}'

# Response: {"answer": "Paris", "confidence": 0.95}
```

#### Guidance - Template-Based
```bash
curl -X POST "http://localhost:8080/v2/models/guidance-example/infer" \
  -H "Content-Type: application/json" \
  -d '{"inputs": [
    {"name": "prompt", "shape": [1], "datatype": "BYTES", "data": ["Why is the sky blue?"]},
    {"name": "template_name", "shape": [1], "datatype": "BYTES", "data": ["chain_of_thought"]}
  ]}'

# Response: {"reasoning": {...}, "answer": "...", "confidence": "high"}
```

#### LMQL - Query-Based
```bash
curl -X POST "http://localhost:8080/v2/models/lmql-example/infer" \
  -H "Content-Type: application/json" \
  -d '{"inputs": [
    {"name": "prompt", "shape": [1], "datatype": "BYTES", "data": ["What is 15 + 27?"]},
    {"name": "query_type", "shape": [1], "datatype": "BYTES", "data": ["scripted"]}
  ]}'

# Response: {"type": "math problem", "steps": [...], "result": "42"}
```

#### Jsonformer - Simple JSON
```bash
curl -X POST "http://localhost:8080/v2/models/jsonformer-example/infer" \
  -H "Content-Type: application/json" \
  -d '{"inputs": [
    {"name": "prompt", "shape": [1], "datatype": "BYTES", "data": ["Tell me about Marie Curie"]},
    {"name": "schema_name", "shape": [1], "datatype": "BYTES", "data": ["person"]}
  ]}'

# Response: {"name": "Marie Curie", "occupation": "physicist", ...}
```

#### Instructor - Pydantic Validation
```bash
curl -X POST "http://localhost:8080/v2/models/instructor-example/infer" \
  -H "Content-Type: application/json" \
  -d '{"inputs": [
    {"name": "prompt", "shape": [1], "datatype": "BYTES", "data": ["Extract entities: Apple was founded by Steve Jobs in California."]},
    {"name": "schema_name", "shape": [1], "datatype": "BYTES", "data": ["extraction"]}
  ]}'

# Response: {"entities": [{"name": "Apple", "type": "organization"}, ...]}
```

## Model Structure

Each model follows the Triton Python backend structure:

```
<model-name>/
├── config.pbtxt      # Triton model configuration
└── 1/
    └── model.py      # Python backend implementation
```

### config.pbtxt

All models share similar configuration:
- **Inputs**: `prompt` (required), `max_tokens`, `temperature`, and model-specific options
- **Outputs**: `generated_text`, `token_count`
- **Instance**: CPU by default (change to `KIND_GPU` for GPU)

### model.py

Each implementation includes:
- `initialize()` - Load model and set up library
- `execute()` - Run inference with constraints
- `finalize()` - Clean up resources

## Available Schemas/Templates

### outlines-example
| schema_name | Description |
|-------------|-------------|
| `qa` | Question-answering with confidence |
| `entity` | Named entity extraction |
| `sentiment` | Sentiment analysis |
| `regex_phone` | Phone number extraction (regex) |

### guidance-example
| template_name | Description |
|---------------|-------------|
| `qa` | Simple Q&A with JSON output |
| `chain_of_thought` | Step-by-step reasoning |
| `classification` | Multi-class classification |

### lmql-example
| query_type | Description |
|------------|-------------|
| `basic` | Simple generation with stop condition |
| `constrained` | Type constraints (INT, length) |
| `scripted` | Python control flow in query |

### jsonformer-example
| schema_name | Description |
|-------------|-------------|
| `qa` | Question-answering |
| `person` | Person information extraction |
| `product` | Product description |
| `list_items` | List generation |

### instructor-example
| schema_name | Description |
|-------------|-------------|
| `qa` | Q&A with confidence and reasoning |
| `extraction` | Named entity extraction |
| `analysis` | Sentiment analysis |

## Choosing a Library

| Use Case | Recommended |
|----------|-------------|
| Production JSON APIs | **Outlines** - Most mature, best performance |
| Complex output templates | **Guidance** - Intuitive DSL, flexible |
| Declarative constraints | **LMQL** - Clean query syntax |
| Simple JSON only | **Jsonformer** - Minimal overhead |
| OpenAI API compatibility | **Instructor** - Designed for APIs |
| Custom grammars | **Outlines** - CFG support |
| Multi-turn conversations | **Guidance** - State preservation |

## Performance Notes

- **Outlines**: Compiles schemas into efficient token masks. Initial compilation has overhead, but subsequent generations are fast.
- **Guidance**: Moderate overhead from template parsing. Good for complex control flow.
- **LMQL**: Async execution with constraint checking. Good for complex queries.
- **Jsonformer**: Minimal overhead since structure is deterministic.
- **Instructor**: May require retries, so latency can vary.

## Extending These Examples

To add your own schemas/templates:

1. **Outlines**: Add Pydantic models to `self.schemas` and build generators in `initialize()`
2. **Guidance**: Add template functions in `_run_template()`
3. **LMQL**: Add decorated async query functions
4. **Jsonformer**: Add JSON Schema definitions to `self.schemas`
5. **Instructor**: Add Pydantic models to `self.schemas`

## References

- [Outlines Documentation](https://outlines-dev.github.io/outlines/)
- [Guidance Documentation](https://guidance-ai.github.io/guidance/)
- [LMQL Documentation](https://lmql.ai/docs/)
- [Jsonformer GitHub](https://github.com/1rgs/jsonformer)
- [Instructor Documentation](https://python.useinstructor.com/)
- [Triton Python Backend](https://github.com/triton-inference-server/python_backend)
