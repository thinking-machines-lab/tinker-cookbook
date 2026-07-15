# Inkling scripts

Minimal scripts for sampling Inkling with reasoning-effort, audio, and image
inputs. They require the `inkling` extra:

```bash
uv pip install 'tinker-cookbook[inkling]'
```

For setup, supported inputs, reasoning-effort behavior, and API guidance, see
the canonical [Using Inkling documentation](https://tinker-docs.thinkingmachines.ai/cookbook/inkling/).

## Reasoning effort

```bash
python -m tinker_cookbook.scripts.inkling.sample_reasoning
python -m tinker_cookbook.scripts.inkling.sample_reasoning \
    efforts='[0.0,0.01,0.3,0.6,0.9,0.99]'
```

See the [reasoning-effort guide](https://tinker-docs.thinkingmachines.ai/cookbook/inkling/thinking-effort/).

## Audio

```bash
python -m tinker_cookbook.scripts.inkling.sample_audio
python -m tinker_cookbook.scripts.inkling.sample_audio message_format=chat
```

See the [Inkling audio guide](https://tinker-docs.thinkingmachines.ai/cookbook/inkling/audio/).

## Images

```bash
python -m tinker_cookbook.scripts.inkling.sample_vision
python -m tinker_cookbook.scripts.inkling.sample_vision message_format=chat
```

See the [Inkling images guide](https://tinker-docs.thinkingmachines.ai/cookbook/inkling/images/).
