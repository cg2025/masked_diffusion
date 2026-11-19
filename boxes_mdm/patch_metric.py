# usage: python patch_metric.py <path to diffusion-vs-ar>
# Idempotent: safe to run more than once.
import sys
p = sys.argv[1] + "/src/llmtuner/tuner/core/metric.py"
s = open(p).read()
if "elif 'boxes' in data_name" in s:
    print("already patched", p)
    sys.exit(0)
old = "        elif 'path' in data_name:\n"
new = '''        elif 'boxes' in data_name:  ## Kim & Schuster entity tracking: exact set match of box contents
            def _items(x):
                x = x.strip()
                if x == 'is empty':
                    return frozenset()
                if not x.startswith('contains '):
                    return None
                return frozenset(i.replace('the ', '', 1).strip() for i in x[len('contains '):].split(' and '))
            _p, _l = _items(pred), _items(label)
            score_dict["acc"].append(_p is not None and _p == _l)

'''
assert s.count(old) == 1, "unexpected metric.py layout (upstream changed?)"
open(p, "w").write(s.replace(old, new + old))
print("patched", p)
