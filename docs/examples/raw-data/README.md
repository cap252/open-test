# Raw data examples

These CSV tables retain measurements exported from saved GPU evaluations with `reproduce report --table-profile opennpu_raw`. Recovery has been recalculated from their retained `metrics_json` using the current report formula. No inference was rerun. They illustrate the [15-column format](../../reporting.md#the-15-column-table).

- `raw_data.csv`: two completed conditions from one full run on all 5,000 COCO images, FP32 and Sym/PT PoT AdaRound of the additional model YOLOv9-S.
- `raw_data.partial.csv`: ResNet-18 FP32 and Asym/PC PoT RTN from a three-sample run.
- [provenance.json](provenance.json): original run and package provenance, the table revision, and current CSV file hashes. The source runs were not revalidated during the table revision.

The two tables come from separate runs and do not share FP32 denominators. The report commands normally also write `raw_data.metadata.json` and the headers of empty groups, which are omitted here.
