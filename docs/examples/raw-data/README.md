# Raw data examples

These tables were exported from saved GPU evaluations with `reproduce report --table-profile opennpu_raw`. They illustrate the [15-column format](../../reporting.md#the-15-column-table).

- `raw_data.csv` and `raw_data.tsv`: two completed conditions from one full run on all 5,000 COCO images, FP32 and Sym/PT PoT AdaRound of the additional model YOLOv9-S.
- `raw_data.partial.csv` and `raw_data.partial.tsv`: ResNet-18 FP32 and Asym/PC PoT RTN from a three-sample run.
- [provenance.json](provenance.json): hashes of the input run, prepared package, selection and output files.

The two tables come from separate runs and do not share FP32 denominators. The report commands normally also write `raw_data.metadata.json` and the headers of empty groups, which are omitted here.
