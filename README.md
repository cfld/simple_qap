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

```
$ OMP_NUM_THREADS=1 python test.py
{
  "sq_time": 4.903386354446411,
  "faq_time": 0.009653806686401367,
  "two_time": 0.7653300762176514,
  "sq_score": 15978,
  "faq_score": 16060,
  "two_score": 16276
}

$ OMP_NUM_THREADS=24 python test.py
{
  "sq_time": 0.3665273189544678,
  "faq_time": 0.014138460159301758,
  "two_time": 0.32042860984802246,
  "sq_score": 15918,
  "faq_score": 16060,
  "two_score": 16476
}
```