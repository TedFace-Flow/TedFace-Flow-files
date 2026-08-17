# Model files

Large model weights are distributed through the repository's GitHub Releases page and
are not committed as ordinary Git objects.

Expected local files:

```text
models/tedface_flow_final.pt
models/tedface_flow_orbital_domain.pt
models/maisi/autoencoder_v1.pt
models/maisi/diff_unet_3d_rflow-ct.pt
```

`tedface_flow_final.pt` is required for fixed-atlas inference. The orbital-domain
checkpoint initializes cross-modal phase 1 and is required only when reproducing the
four-phase training schedule. The MAISI-v2 weights are not redistributed by this
repository; obtain them from their official source and retain the original license and
attribution.

Final inference checkpoint checksum:

```text
SHA-256: 153768cb0cb837e8ddccf17e54a6cbe5b1f02c3cb4dff50c0a2c4ee21c10fce7
```
