import sys, pickle, time
sys.path.insert(0, '.')
from qprgnn.parse import iter_samples
for t in sys.argv[1:]:
    t0 = time.time(); S = list(iter_samples('data_traces/' + t))
    name = t.replace('/', '_')
    with open(f'cache/{name}.pkl', 'wb') as f: pickle.dump(S, f, protocol=4)
    print(name, len(S), 'samples', f'{time.time()-t0:.0f}s', flush=True)
