Extension for Forge derived webUIs for StableDiffusion, that support the [Anima](https://huggingface.co/circlestone-labs/Anima) model and the LLLite controlnets by Kohya:
*    Specifically, Forge Neo and my own fork. Currently tested only on my own fork, and already included in that repo.

Adds five new lineart preprocessors for ControlNet:
* `lineart_anime_inverted` is simply a modified `lineart_anime_denoised` using the MangaLine (erika.pth) model
    * model probably already exists, but if not it will be automatically downloaded: 164MB
* `AniLines basic` and `AniLines detail` are newer models by https://github.com/zhenglinpan
    * see [AniLines-Anime-Lineart-Extractor](https://github.com/zhenglinpan/AniLines-Anime-Lineart-Extractor/)
    * models automatically downloaded: 66MB each
* `lineart_xDoG` and `lineart_xDoG_inverted` are basic but effective Difference of Gaussians filters (no model used)

With the exception of `lineart_xDoG`, these produce black-line-on-white-background output - opposite to the *traditional* preprocessor output. The first three of the five are the preprocessors used in training the [controlnet for Anima](https://huggingface.co/kohya-ss/Anima-LLLite)
