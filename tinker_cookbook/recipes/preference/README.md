# Learning from Preferences

Many applications involve learning from preferences beyond scalar rewards. We provide a few examples here:

1. [Shorter](./shorter/): we introduce the `PairwisePreferenceRLDatasetBuilder` abstraction and walk through a simple example that trains a model to generate shorter responses.
2. [RLHF](./rlhf/): we walk through the standard RLHF pipeline from [1, 2]. This pipeline involves three stages: supervised fine-tuning, reward model learning, and reinforcement learning.
3. [DPO](./dpo/): we optimize for human preferences using the Direct Preference Optimization algorithm [3], which requires a custom loss function.

**References:**
1. Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion, J., Jones, A., Chen, A., Goldie, A., Mirhoseini, A., McKinnon, C., Chen, C., Olsson, C., Olah, C., Hernandez, D., Drain, D., Ganguli, D., Li, D., Tran-Johnson, E., Perez, E., Kerr, J., Mueller, J., Ladish, J., Landau, J., Ndousse, K., Lukošiūtė, K., Lovitt, L., Sellitto, M., Elhage, N., Schiefer, N., Mercado, N., ... Kaplan, J. (2022). Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv. https://arxiv.org/abs/2204.05862
2. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P. F., Leike, J., & Lowe, R. (2022). Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35, 27730-27744. https://arxiv.org/abs/2203.02155
3. Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct preference optimization: Your language model is secretly a reward model. arXiv. https://arxiv.org/abs/2305.18290
