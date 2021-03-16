# simple_qap

- Local-search based quadratic assignment problem (QAP) solver w/ Python bindings.
- Parallelism via OpenMP.
- Parameters to control tradeoff between runtime and solution quality.

## Usage

See `./run.sh`

## Notes

- Running w/ `popsize=1` and `piter=1` should be a roughly equivalent algorithm to
```
scipy.optimize.quadratic_assignment(A, B, method='2opt')
```
However, it should be substantially faster (>10x)

- Running w/ `popsize > 1`, `piter > 1` and multiple threads should really dominate `scipy`'s implementation.

## Example

```bash
# nug30.dat / piter=32 / popsize=24
# optimal solution: 6124

$ OMP_NUM_THREADS=24 python test.py
{
  "sq_time"   : 0.08106613159179688,
  "faq_time"  : 0.0065462589263916016,
  "two_time"  : 0.11268234252929688,
  "sq_score"  : 6128,
  "faq_score" : 6290,
  "two_score" : 6336
}
```
