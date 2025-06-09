# From Objects to Anywhere: A Holistic Benchmark for Multi-level Visual Grounding in 3D Scenes
[Tianxu Wang](https://github.com/TXWang98)<sup>1</sup>, [Zhuofan Zhang](https://github.com/zfzhang-thu)<sup>1,2</sup>, [Ziyu Zhu](https://github.com/zhuziyu-edward)<sup>1,2</sup>, [Yue Fan](https://github.com/YueFan1014)<sup>1</sup>, [Jing Xiong](https://github.com/Aurora-Xiong/)<sup>1,3</sup>, [Pengxiang Li](https://github.com/Pengxiang-Li)<sup>1,4</sup>, [Xiaojian Ma](https://jeasinema.github.io/)<sup>1</sup>, [Qing Li](https://liqing.io/)<sup>1, *</sup>

*: corresponding author

<sup>1</sup> BIGAI, <sup>2</sup>Tsinghua University, <sup>3</sup>Peking University, <sup>4</sup>Beijing Institute of Technology


<p align="center">
  <a href="https://arxiv.org/abs/2506.04897">
    <img src="https://img.shields.io/badge/arXiv-✏️-black?style=for-the-badge&logoColor=white" alt="arXiv">
  </a>
  <a href="https://github.com/anywhere-3d/Anywhere3D">
    <img src="https://img.shields.io/badge/Code-GitHub-black?style=for-the-badge&logo=github&logoColor=white" alt="Code">
  </a>
  <a href="https://huggingface.co/datasets/txwang98/Anywhere3D">
    <img src="https://img.shields.io/badge/Data-database-black?style=for-the-badge&logo=postgresql&logoColor=white" alt="Data">
  </a>
  <a href="https://anywhere3d-viewer-webpage.onrender.com/apps/meshviewer/datasetname=arkitscene_valid&scene_id=scene0004_00">
    <img src="https://img.shields.io/badge/Annotator-interface-black?style=for-the-badge&logo=visual-studio-code&logoColor=white" alt="Annotator">
  </a>
</p>


## 📰 News

- 📅 **2025/05/26** Released the Human Annotation Interface Demo, supporting four scenes from ScanNet, MultiScan, 3RScan, ARKitScenes. Click [Here](https://anywhere3d-viewer-webpage.onrender.com/apps/meshviewer/datasetname=arkitscene_valid&scene_id=scene0004_00) to play around and further tutorial.
- 📄 **2025/06/04** Paper submitted to arXiv: [Anywhere3D](https://arxiv.org/abs/2506.04897)



## 🧠 Abstract
<details>
<summary><strong>Abstract</strong> (click to expand)</summary>

3D visual grounding has made notable progress in localizing objects within complex 3D scenes. However, grounding referring expressions beyond objects in 3D scenes remains unexplored. In this paper, we introduce Anywhere3D-Bench, a holistic 3D visual grounding benchmark consisting of 2,632 referring expression-3D bounding box pairs spanning four different grounding levels: human-activity areas, unoccupied space beyond objects, objects in the scene, and fine-grained object parts.

We assess a range of state-of-the-art 3D visual grounding methods alongside large language models (LLMs) and multimodal LLMs (MLLMs) on Anywhere3D-Bench. Experimental results reveal that space-level and part-level visual grounding pose the greatest challenges: space-level tasks require a more comprehensive spatial reasoning ability, for example, modeling distances and spatial relations within 3D space, while part-level tasks demand fine-grained perception of object composition.

Even the best performance model, OpenAI o4-mini, achieves

