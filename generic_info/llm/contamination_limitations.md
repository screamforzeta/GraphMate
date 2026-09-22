# Contamination Limitations

A3/A4 were trained on the Lichess training population and evaluated on the frozen Lichess test split.

Local LLMs were pretrained on broad external corpora. They may have seen chess material, Lichess-derived content, or classic puzzle material during pretraining.

The Lichess test comparison is useful but not sufficient for a contamination-free external-validity claim.

The future held-out classic benchmark is also useful, but classic problems may exist online and may also be present in LLM training data.

Report this limitation explicitly in final analysis.
