# Theory: why decomposing the per-hop delay matters

## Setting
A FIFO output port with capacity C (bit/s) carries aggregate traffic λ (bit/s)
made of flows with different mean packet sizes. Let s_f be the mean packet size
of a tagged flow f, s̄ the packet-weighted mean packet size of all traffic on the
port, and ρ = λ / C. Packet arrivals are Poisson; service times are size / C.

## Proposition 1 (shared waiting, private transmission)
Under FIFO service the waiting time W of an arriving packet does not depend on
its own size (PASTA + FIFO). The mean per-hop delay of flow f is therefore

  D_f = W + s_f / C.

With exponentially distributed service (M/M/1 equivalent with mean s̄),
W = (s̄ / C) · ρ / (1 − ρ), so

  D_f = s_f / C + (s̄ / C) · ρ / (1 − ρ)        (decomposed prior; with finite buffers, M/M/1/K)

The prior used by M/M/1-style residual models, D_f^old = (s_f / C) / (1 − ρ),
treats the flow's own packet size as the mean service time of the whole queue.
Its ratio to the decomposed delay is

  D_f^old / D_f = s_f / (ρ · s̄ + (1 − ρ) · s_f).

Hence:
* The two coincide only when s_f = s̄ (homogeneous packet sizes) or ρ → 0.
* Flows with large packets (s_f > s̄) are over-predicted, flows with small
  packets are under-predicted, and the error grows with ρ. As ρ → 1 the ratio
  tends to s_f / s̄.
* The correction a residual model must learn on top of the old prior therefore
  depends on s_f / s̄. That is a **non-local** quantity: it requires the packet
  sizes of every other flow sharing the port. With the decomposed prior this
  dependence is removed, so the remaining correction (burstiness, scheduling)
  is the same for all flows on a port. That is easier to learn and to transfer.

For general service distributions (M/G/1) the Pollaczek–Khinchine formula gives
W = λ_p E[S²] / (2(1 − ρ)), which again does not depend on s_f; the decomposition
D_f = W + s_f / C is exact for any FIFO M/G/1 queue.

## Where this is tested
* Prior ablation (`results/analysis/prior_ablation.txt`): with the same backbone, replacing
  tx/(1 − ρ) by the decomposed prior s_f/C + W lowers the held-out error; finite buffers
  (M/M/1/K) help a little more.
* Error by packet-size mismatch: `results/analysis/error_by_size_mismatch.txt`.
