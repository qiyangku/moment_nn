
Sample usage for evaluating the moment activation
```
import mnn_core.maf

maf = MomentActivation()
ubar = np.array([2, 3])
sbar = np.array([2, 5])

u = maf.mean(ubar,sbar)
s = maf.std(ubar,sbar)
chi = maf.chi(ubar,sbar)

```

Sample usage for evaluating Dawson-like functions

```
import mnn_core.fast_dawson

ds1 = Dawson1()
ds2 = Dawson2()    

x = np.arange(-3, 1.2, 0.01)
H = ds2.int_fast(x)
h = ds2.dawson2(x)
G = ds1.int_fast(x)
g = ds1.dawson1(x)
```
