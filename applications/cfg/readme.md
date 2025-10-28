# Debiasing CFG with CREPE

## 🛠️Env and code preparation
Our code is based on the EDM2 repo directly.  Therefore, please clone the repo from [https://github.com/NVlabs/edm2/tree/main](https://github.com/NVlabs/edm2/tree/main) and install the environment from their repo.


### 👉Tips for installing ImageReward 

Skip this section if you have ImageReward correctly installed; otherwise, here are some tips.

We found that installing ImageReward directly (from either pip or the repo) always ran into the issue that the `requirements.txt` file required `transformers>=4.27.4`. However, these versions of `transformers` do not support the imports used in `ImageReward/ImageReward/models/BLIP/med.py`. Namely, the code block:

```
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
```

Therefore, to fix this issue, we had to replace the above code with the new code below:

```
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
```

and then pip install the repo locally.


## ✨Sampling
```
python crepe.py ---image_net 64
```


