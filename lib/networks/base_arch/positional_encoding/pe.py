import torch


class FreqEncoder:
    default_encoder_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'num_freqs': 10,  # 3 + 3*128= 387
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    def __init__(self, **kwargs):
        self.kwargs = {**self.default_encoder_kwargs, **kwargs}
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        # max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        max_freq = N_freqs - 1

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
