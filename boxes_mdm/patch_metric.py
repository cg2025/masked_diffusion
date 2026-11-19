# usage: python patch_metric.py ~/diffusion-vs-ar
import sys
p = sys.argv[1] + "/src/llmtuner/tuner/core/metric.py"
s = open(p).read()
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
assert s.count(old) == 1 and "'boxes'" not in s, "already patched or unexpected file"
open(p, "w").write(s.replace(old, new + old))
print("patched", p)
