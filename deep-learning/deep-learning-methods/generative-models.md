# Generative Models

Generative models learn the distribution of data and can sample new examples. Post-2020, **diffusion models** dominate image and video; **autoregressive** and **latent** variants power text and multimodal generation; **GANs** remain relevant with modern improvements.

---

## Diffusion models

**Idea:** Gradually add noise to data (forward process), then train a network to reverse the process (denoising). Sampling: start from noise and iteratively denoise.

**DDPM (Denoising Diffusion Probabilistic Models):**  
- **Forward:** \(q(x_t | x_{t-1})\): add Gaussian noise over \(T\) steps until \(x_T \approx \mathcal{N}(0, I)\).  
- **Reverse:** \(p_\theta(x_{t-1}|x_t)\): neural network predicts noise or mean; often parameterized to predict \(\epsilon\) (noise) at step \(t\).  
- **Training:** For random \(t\) and noise \(\epsilon\), input \(x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon\), loss = \(\|\epsilon - \epsilon_\theta(x_t, t)\|^2\).  
- **Sampling:** From \(x_T \sim \mathcal{N}(0,I)\), run \(T\) denoising steps to get \(x_0\).

**Latent diffusion (Stable Diffusion):** Run diffusion in a **latent space** (e.g. VAE encoder output), not pixel space. Reduces compute; conditioning (e.g. text via cross-attention) is added in the denoising UNet. Text-to-image: encode text with a T5/CLIP text encoder; inject into the UNet via cross-attention.

**Text-to-image:** Models like DALL·E 2, Stable Diffusion, Imagen: condition the diffusion model on text embeddings; often classifier-free guidance to boost alignment with the prompt.

**Text-to-video:** Extend to temporal dimension (3D UNet or diffusion over video latents); examples: Runway, Sora-style models.

---

## GANs and modern improvements

**GAN:** Generator \(G\) maps noise to data; discriminator \(D\) classifies real vs generated. Train with adversarial loss: \(D\) maximizes correct classification, \(G\) minimizes \(\log(1-D(G(z)))\) or maximizes \(\log D(G(z))\).

**StyleGAN (StyleGAN2):** Style injection at multiple layers; mapping network from latent to “style” space; improves control and quality. **StyleGAN2/3:** Refinements for artifacts and training stability.

**Modern GAN improvements:** Better training (TTUR, spectral norm, projection discriminator), conditional GANs, and use in domains where diffusion is still expensive (e.g. some real-time or low-compute settings).

---

## Autoregressive generative models

- **Language models:** Next-token prediction; GPT, LLaMA, etc. **Image:** PixelCNN, Image GPT (generate pixels in order). **Multimodal:** Autoregressive over tokens (text + image tokens in one sequence).
- **Pros:** Simple objective, flexible. **Cons:** Sequential sampling can be slow; long-range dependencies.

---

## Quick revision

- **Diffusion:** Forward = add noise; reverse = learn denoising; sample by denoising from noise. **Latent diffusion:** Diffusion in VAE latent space; text conditioning via cross-attention (Stable Diffusion). **GANs:** Generator vs discriminator; StyleGAN for high-quality controllable images. **Autoregressive:** Next-token or next-pixel; standard for LLMs and some image models.
