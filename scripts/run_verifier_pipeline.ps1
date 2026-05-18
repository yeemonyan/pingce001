param(
    [string]$InputPath = "data/raw/train.json",
    [string]$TrainIds = "data/splits/train_ids.json",
    [string]$DevIds = "data/splits/dev_ids.json",
    [string]$DevPrompts = "outputs/dev_prompts.jsonl",
    [string]$OutputDir = "outputs",
    [int]$Seed = 2026,
    [int]$OversamplePriority = 1,
    [ValidateSet("base", "domain_hint")]
    [string]$InstructionVariant = "base",
    [switch]$CandidateOnly
)

$ErrorActionPreference = "Stop"

$buildArgs = @(
    "scripts\build_verifier_data.py",
    "--input", $InputPath,
    "--train-ids", $TrainIds,
    "--dev-ids", $DevIds,
    "--output-dir", $OutputDir,
    "--report", (Join-Path $OutputDir "verifier_data_report.json"),
    "--compat-output-names",
    "--instruction-variant", $InstructionVariant,
    "--oversample-priority", "$OversamplePriority"
)
if ($CandidateOnly) {
    $buildArgs += "--candidate-only"
}
python @buildArgs

python scripts\audit_verifier_data.py `
    --input (Join-Path $OutputDir "sft_valid_verifier.jsonl") `
    --output (Join-Path $OutputDir "verifier_data_audit.json") `
    --seed $Seed `
    --sample-count 10 `
    --domain-count 5 `
    --language-count 10 `
    --long-count 10

$inferArgs = @(
    "scripts\infer_verifier.py",
    "--input", $DevPrompts,
    "--backend", "mock",
    "--output", (Join-Path $OutputDir "dev_verifier_mock_predictions.jsonl"),
    "--option-output", (Join-Path $OutputDir "dev_verifier_mock_option_predictions.jsonl"),
    "--report", (Join-Path $OutputDir "dev_verifier_mock_eval.json"),
    "--instruction-variant", $InstructionVariant
)
if ($CandidateOnly) {
    $inferArgs += "--candidate-only"
}
python @inferArgs

python -m unittest discover -s tests -v
