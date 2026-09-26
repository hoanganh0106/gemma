# P123 K2 residual direct HBM probe

Starting from P122, the probe attempted to feed the HBM residual view directly
to `device.sub.begin`, removing the residual HBM-to-DM transfer.

SDK 0.8.1 rejects this at type checking:

```text
expected DmTensorView, found HbmTensorView
```

The Sub context accepts DM tensors only. The 616-cycle residual materialization
cannot be removed through a direct HBM read in this API.
