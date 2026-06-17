# Final Submitted Versions

This repo has been cleaned so the main task folders contain the submitted
runtime sources, not old experiments or alternate variants.

| Task | Submitted image | Source kept in repo |
|---|---|---|
| ASR | `nobrainnohack-asr:finals` (`sha256:b7ab13213ad7...`) | `asr/src`, `asr/postprocess` |
| NLP | `nobrainnohack-nlp:finals` (`sha256:21d7a48948cb...`) | `nlp/src` |
| CV | `nobrainnohack-cv:finals` (`sha256:3e33d543c301...`) | `cv/src` |
| Noise | `nobrainnohack-noise:finals` (`sha256:8906fbfe3a5a...`) | `noise/src` |
| AE | `nobrainnohack-ae:finals` (`sha256:076a66ee2055...`) | `ae/src` scripted runtime, `AE_STRATEGY=balanced_extreme_opening` |
| Surprise | `nobrainnohack-surprise:latest` (`sha256:3f65d57ecbda...`) | `shizhen-suprise-server/opponents/shizhen-yilin-frozen` |

Large runtime artifacts such as model weights remain ignored by Git. The
submitted Docker images are the source of truth for those binaries.
