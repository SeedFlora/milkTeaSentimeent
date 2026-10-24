"""Verify this bounded revision release using only Python's standard library.

This reproduces retained prediction metrics and summaries; it does not retrain
models or regenerate unavailable review texts, transformer predictions or weights.
"""
from pathlib import Path
import csv, hashlib, json, math, statistics

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def auc(y, scores):
    ranked = sorted(zip(scores, y))
    rank_sum = 0.0
    i = 0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        rank_sum += sum(label for _, label in ranked[i:j]) * ((i + 1 + j) / 2)
        i = j
    positives = sum(y)
    negatives = len(y) - positives
    return (rank_sum - positives*(positives+1)/2)/(positives*negatives)

def metrics(rows):
    y = [int(x['label_id']) for x in rows]
    p = [int(x['predicted_label_id']) for x in rows]
    scores = [float(x['positive_score']) for x in rows]
    class_f1 = []
    for label in (0, 1):
        tp = sum(a == b == label for a,b in zip(y,p))
        fp = sum(a != label and b == label for a,b in zip(y,p))
        fn = sum(a == label and b != label for a,b in zip(y,p))
        class_f1.append(2*tp/(2*tp+fp+fn))
    return {'macro_f1':statistics.mean(class_f1),
            'accuracy':sum(a == b for a,b in zip(y,p))/len(y),
            'auc':auc(y,scores)}

def main():
    manifest = json.loads((ROOT/'MANIFEST.json').read_text())
    for file in manifest['files']:
        p = ROOT/file['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == file['sha256'], p
    runs = read(ROOT/'results/main_individual_runs.csv')
    assert len(runs) == 80
    verified = 0
    for run in runs:
        name = run['released_prediction_file']
        if not name:
            continue
        rows = read(ROOT/name)
        assert len(rows) == int(run['test_n'])
        assert set(rows[0]) == {'example_id','label_id','predicted_label_id','positive_score'}
        assert len({x['example_id'] for x in rows}) == len(rows)
        for key,value in metrics(rows).items():
            assert abs(value-float(run[key])) < 1e-10, (run['model'],key)
        verified += 1
    summary = read(ROOT/'results/recomputed_model_summary.csv')
    for model in summary:
        group = [r for r in runs if r['model'] == model['model']]
        assert len(group) == 5
        for key in ['macro_f1','accuracy','auc','fit_plus_training_features_sec','serialized_full_pipeline_bytes']:
            values = [float(x[key]) for x in group]
            for suffix,value in [('mean',statistics.mean(values)),('std',statistics.stdev(values))]:
                assert math.isclose(value,float(model[key+'_'+suffix]),rel_tol=1e-10,abs_tol=1e-10), (model['model'],key,suffix)
    assert verified == 30
    print('PASS: file hashes; 30 CPU prediction runs; 80 retained aggregate run records; 16 model summaries.')
    print('Scope: no transformer predictions or weights; no raw review data; no archived supplement.')

if __name__ == '__main__':
    main()
