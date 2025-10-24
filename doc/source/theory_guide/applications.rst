.. _neural_op_applications:

============================
Neural Operator Applications
============================

This guide presents examples of neural operator applications across scientific and 
engineering domains, demonstrating their transformative impact on computational 
science and real-world problem solving.

Introduction
============

Neural operators have been successfully applied to a wide variety of scientific and 
engineering problems, from fluid dynamics and climate modeling to materials 
science and biomedical applications. Their ability to learn mappings between 
function spaces makes them particularly well-suited for problems involving 
continuous physical phenomena governed by partial differential equations.

The examples presented here span multiple scales, from molecular dynamics to climate systems, 
and demonstrate the versatility of neural operators in handling complex, 
multi-physics problems that were previously computationally prohibitive or 
intractable with traditional methods. This is not an exhaustive list, but rather 
a selection of representative applications that showcase the potential of neural operators.

.. raw:: html

   <div style="margin-top: 3em;"></div>

Fluid and Solid Mechanics
=========================

**Computational Mechanics Applications**

Neural operators have fostered significant advancements in computational mechanics, 
including modeling porous media, fluid mechanics, and solid mechanics [2]_ [3]_. 
They offer substantial speedups over traditional numerical solvers while achieving 
competitive accuracies and expanding their features [4]_ [5]_.

**Turbulent Flow Modeling**

Fourier Neural Operators (FNOs) constitute the first machine learning-based method 
to successfully model turbulent flows with zero-shot super-resolution capabilities [6]_. 
This breakthrough enables researchers to predict complex fluid behavior at resolutions 
that were previously computationally prohibitive. The ability to generalize across 
different Reynolds numbers and flow geometries makes FNOs particularly valuable for 
industrial applications where traditional computational fluid dynamics methods would 
require extensive computational resources.

**Stabilization Techniques**

Sobolev losses and dissipativity-inducing regularization terms are effective in 
stabilizing long autoregressive rollouts for highly turbulent flows [7]_. These 
techniques ensure that neural operators maintain physical consistency over extended 
time periods, which is crucial for applications such as weather forecasting and 
climate modeling where long-term stability is essential.

**Large-Scale Simulations**

Neural operators have also been used in large eddy simulations of three-dimensional 
turbulence [5]_ and to learn the stress-strain fields in digital composites [8]_. 
These applications demonstrate the versatility of neural operators in handling 
complex multi-physics problems that involve multiple length and time scales. The 
ability to learn from high-fidelity simulation data and then rapidly predict 
solutions for new configurations makes neural operators invaluable for design 
optimization and uncertainty quantification.

**Probabilistic Modeling**

Finally, neural operators have been used in combination with diffusion models on function 
spaces to learn distributions over solutions when given sparse or noisy observations [9]_. 
This capability is particularly important for real-world applications where data may be 
incomplete or uncertain, enabling robust predictions even in the presence of measurement 
noise or missing information.

.. raw:: html

   <div style="margin-top: 3em;"></div>

Nuclear Fusion and Plasma Physics
==================================

**Magnetohydrodynamic Simulations**

Neural operators have been used to accelerate magnetohydrodynamic (MHD) simulations 
for plasma evolution both from state and camera data [10]_ [11]_. This represents a 
significant advancement in fusion energy research, where understanding plasma behavior 
is crucial for developing sustainable fusion reactors. The ability to rapidly simulate 
complex plasma dynamics enables researchers to explore a wider range of operating 
conditions and design parameters than would be possible with traditional methods.

**Plasma Instability Analysis**

Instabilities arising in long-term rollouts using neural operators for plasma evolution 
have been studied [12]_, together with the potential of learning across different MHD 
simulation codes, data fidelities, and subsets of state variables. This research is 
critical for understanding how plasma instabilities develop and propagate in fusion 
devices, which directly impacts the efficiency and safety of fusion reactors.

**Tokamak Discharge Classification**

Furthermore, neural operators have been used for labeling the confinement states of 
tokamak discharges [13]_. This application is particularly important for real-time 
control of fusion experiments, where rapid classification of plasma states can help 
operators make informed decisions about experimental parameters and safety protocols. 
The ability to process high-dimensional plasma data in real-time represents a major 
step forward in fusion energy research and development.

.. raw:: html

   <div style="margin-top: 3em;"></div>

Geoscience and Environmental Engineering
========================================

**Seismic Wave Propagation and Inversion**

In the geosciences, FNOs and UNOs have been used for seismic wave propagation and 
inversion [14]_ [15]_. These applications are crucial for understanding Earth's 
internal structure and for oil and gas exploration. The ability to rapidly process 
seismic data and invert for subsurface properties enables geophysicists to make 
more informed decisions about resource exploration and geological hazard assessment.

**Earth Surface Movement Modeling**

Extensions of generative models to function spaces have been employed to model earth 
surface movements in response to volcanic eruptions or earthquakes, or subsidence 
due to excessive groundwater extraction [16]_ [17]_. These applications are essential 
for understanding natural hazards and their impact on human populations. The ability 
to predict ground deformation patterns helps in disaster preparedness and mitigation 
planning, particularly in regions prone to seismic activity or volcanic eruptions.

**Multiphase Flow in Porous Media**

Neural operators have also been used to model multiphase flow in porous media, which 
is critical for applications such as contaminant transport, carbon capture and storage, 
hydrogen storage, and nuclear waste storage [18]_ [19]_ [20]_. These applications are 
increasingly important as society seeks to address climate change through carbon 
capture technologies and transition to clean energy sources. The ability to accurately 
model fluid flow in complex geological formations is crucial for ensuring the safety 
and effectiveness of these technologies.

.. raw:: html

   <div style="margin-top: 3em;"></div>

Weather and Climate Forecasting
=================================

**Numerical Weather Prediction**

Versions of FNOs can match the accuracy of physics-based numerical weather 
prediction systems while being orders-of-magnitude faster [4]_ [21]_. This represents 
a paradigm shift in weather forecasting, enabling more frequent and higher-resolution 
forecasts that can better capture local weather phenomena. The speed advantage of 
neural operators allows for ensemble forecasting and rapid updates as new data becomes 
available, which is crucial for severe weather warnings and emergency response.

**Spherical Geometry Handling**

To facilitate stable simulations of atmospheric dynamics on the earth, the spherical 
Fourier neural operator (SFNO) has been introduced to extend FNOs to spherical 
geometries [22]_. This development is particularly important for global weather 
and climate modeling, where the Earth's spherical geometry must be properly 
accounted for to avoid numerical artifacts and maintain physical consistency. 
The SFNO enables accurate modeling of atmospheric circulation patterns and 
large-scale climate phenomena.

**Climate Data Downscaling**

The super-resolution capabilities of FNOs have also been leveraged for downscaling 
of climate data, i.e., predicting climate variables at high resolutions from 
low-resolution simulations [23]_. This capability is essential for regional 
climate impact assessments, where high-resolution local climate information is 
needed for planning and adaptation strategies. The ability to downscale global 
climate models to local scales enables more accurate assessment of climate change 
impacts on specific regions and communities.

**Climate Tipping Points**

Additionally, neural operators have been utilized for tipping point forecasting, 
with potential applications to climate tipping points [24]_. This research is 
critical for understanding the potential for abrupt climate changes and their 
cascading effects on global climate systems. The ability to identify and predict 
climate tipping points could provide early warning systems for catastrophic 
climate changes and inform mitigation strategies.

.. raw:: html

   <div style="margin-top: 3em;"></div>

Medicine and Healthcare
========================

**Medical Imaging Applications**

Neural operators have been used in multiple settings to improve medical imaging, 
such as ultrasound computer tomography [25]_ [26]_ [27]_. These applications represent 
a significant advancement in medical diagnostics, enabling more accurate and 
rapid imaging procedures that can improve patient outcomes and reduce healthcare costs.

**Lung Disease Diagnosis**

As an example, they have been used on radio-frequency data from lung ultrasounds 
to accurately reconstruct lung aeration maps, which can be used for diagnosing 
and monitoring acute and chronic lung diseases [27]_. This application is particularly 
important for respiratory medicine, where early detection of lung conditions can 
significantly improve treatment outcomes. The ability to process ultrasound data 
in real-time enables point-of-care diagnostics in resource-limited settings.

**MRI Reconstruction**

FNOs supplemented with local integral and differential kernels have been used for 
MRI reconstructions [28]_ [29]_. This development is crucial for reducing scan times 
and improving image quality in magnetic resonance imaging. The ability to reconstruct 
high-quality images from undersampled data enables faster and more comfortable 
patient experiences while maintaining diagnostic accuracy.

**Medical Device Design**

Neural operators have also been used to improve the design of medical devices, 
such as catheters with reduced risk of catheter-associated urinary tract infection [30]_. 
This application demonstrates the potential of neural operators in biomedical 
engineering, where understanding fluid dynamics and material properties is crucial 
for designing safer and more effective medical devices.

**Spatial Transcriptomics**

Finally, GNOs have been used for spatial transcriptomics data classification [31]_. 
This application is at the forefront of precision medicine, where understanding 
the spatial organization of gene expression in tissues can provide insights into 
disease mechanisms and potential therapeutic targets. The ability to process 
high-dimensional biological data efficiently enables researchers to explore 
complex biological systems at unprecedented resolution.

.. raw:: html

   <div style="margin-top: 3em;"></div>

Computer Vision
================

Neural operators have been effectively adapted to computer vision tasks. 
They have served as efficient token mixers in vision transformers [32]_, sped up diffusion 
model sampling for faster image and media generation [33]_, and have been applied in 
image classification [34]_ and segmentation [35]_. 

Their ability to handle images at multiple resolutions and integrate with existing deep 
learning methods makes them a versatile tool for vision applications.

.. raw:: html

   <div style="margin-top: 3em;"></div>

References
==========

.. [1] Principled Approaches for Extending Neural Architectures to Function Spaces for Operator Learning,
       Julius Berner, Miguel Liu-Schiaffini, Jean Kossaifi, Valentin Duruisseaux, 
       Boris Bonev, Kamyar Azizzadenesheli, Anima Anandkumar, 2025.
       arXiv:2506.10973. https://arxiv.org/abs/2506.10973

.. raw:: html

   <div style="margin-top: 1em"></div>

.. [2] Learning deep implicit Fourier neural operators (IFNOs) with applications to heterogeneous material modeling,
       Huaiqian You, Quinn Zhang, Colton J Ross, Chung-Hao Lee, Yue Yu, 2022.
       Computer Methods in Applied Mechanics and Engineering, 398, 115296.

.. raw:: html

   <div style="margin-top: 1em"></div>

.. [3] Fourier Neural Operator for Fluid Flow in Small-Shape 2D Simulated Porous Media Dataset,
       A Choubineh, J Chen, DA Wood, F Coenen, F Ma, 2023.
       Algorithms, 16(1), 24.

.. raw:: html

   <div style="margin-top: 1em"></div>

.. [4] FourCastNet: Accelerating Global High-Resolution Weather Forecasting Using Adaptive Fourier Neural Operators,
       Thorsten Kurth, Shashank Subramanian, Peter Harrington, Jaideep Pathak, Morteza Mardani, 
       David Hall, Andrea Miele, Karthik Kashinath, Anima Anandkumar, 2023.
       Proceedings of the Platform for Advanced Scientific Computing Conference.

.. raw:: html

   <div style="margin-top: 1em"></div>

.. [5] Fourier neural operator approach to large eddy simulation of three-dimensional turbulence,
       Zhijie Li, Wenhui Peng, Zelong Yuan, Jianchun Wang, 2022.
       Theoretical and Applied Mechanics Letters, 12(6), 100389.

.. raw:: html

   <div style="margin-top: 1em"></div>

.. [6] Efficient super-resolution of near-surface climate modeling using the Fourier neural operator,
       P Jiang, Z Yang, J Wang, C Huang, P Xue, TC Chakraborty, 2023.
       Journal of Advances in Modeling Earth Systems, 15.

.. raw:: html

   <div style="margin-top: 1em"></div>

.. [7] Learning the stress-strain fields in digital composites using Fourier neural operator,
       Meer Mehran Rashid, Tanu Pittie, Souvik Chakraborty, N.M. Anoop Krishnan, 2022.
       iScience, 25(11), 105452.

.. raw:: html

   <div style="margin-top: 1em"></div>

.. [8] Guided Diffusion Sampling on Function Spaces with Applications to PDEs,
       Jiachen Yao, Abbas Mammadov, Julius Berner, Gavin Kerrigan, Jong Chul Ye, 
       Kamyar Azizzadenesheli, Anima Anandkumar, 2025.
       arXiv:2505.17004.

.. raw:: html

   <div style="margin-top: 1em"></div>

.. [9] Plasma surrogate modelling using Fourier neural operators,
       Vignesh Gopakumar, Stanislas Pamela, Lorenzo Zanisi, Zongyi Li, Ander Gray, 
       Daniel Brennand, Nitesh Bhatia, Gregory Stathopoulos, Matt Kusner, 
       Marc Peter Deisenroth, Anima Anandkumar, 2024.
       Nuclear Fusion, 64(5), 056025.

.. raw:: html

   <div style="margin-top: 1em"></div>

.. [10] Neural-Parareal: Self-improving acceleration of fusion MHD simulations using time-parallelisation and neural operators,
        S.J.P. Pamela, N. Carey, J. Brandstetter, R. Akers, L. Zanisi, J. Buchanan, 
        V. Gopakumar, M. Hoelzl, G. Huijsmans, K. Pentland, T. James, G. Antonucci, 2025.
        Computer Physics Communications, 307, 109391.

.. raw:: html

   <div style="margin-top: 1em"></div>

.. [11] Robust Confinement State Classification with Uncertainty Quantification through Ensembled Data-Driven Methods,
        Yoeri Poels, Cristina Venturini, Alessandro Pau, Olivier Sauter, Vlado Menkovski, 
        the TCV team, the WPTE team, 2025.
        arXiv:2502.17397.

.. raw:: html

   <div style="margin-top: 1em"></div>

.. [12] Seismic wave propagation and inversion with neural operators,
        Yan Yang, Angela F Gao, Jorge C Castellanos, Zachary E Ross, 
        Kamyar Azizzadenesheli, Robert W Clayton, 2021.
        The Seismic Record, 1(3), 126-134.

.. raw:: html

   <div style="margin-top: 1em"></div>

.. [13] Accelerating Time-Reversal Imaging with Neural Operators for Real-time Earthquake Locations,
        Hongyu Sun, Yan Yang, Kamyar Azizzadenesheli, Robert W Clayton, Zachary E Ross, 2022.
        arXiv:2210.06636.

.. raw:: html

   <div style="margin-top: 1em"></div>

.. [14] Generative adversarial neural operators,
        Md Ashiqur Rahman, Manuel A Florez, Anima Anandkumar, Zachary E Ross, 
        Kamyar Azizzadenesheli, 2022.
        arXiv:2205.03017.

.. raw:: html

   <div style="margin-top: 1em"></div>

.. [15] Variational Autoencoding Neural Operators,
        Jacob H Seidman, Georgios Kissas, George J Pappas, Paris Perdikaris, 2023.
        arXiv:2302.10351.

.. raw:: html

   <div style="margin-top: 1em"></div>

.. [16] Real-time high-resolution CO2 geological storage prediction using nested Fourier neural operators,
        Gege Wen, Zongyi Li, Qirui Long, Kamyar Azizzadenesheli, Anima Anandkumar, 
        Sally M Benson, 2023.
        Energy Environ. Sci., 16(4), 1732-1741.

.. raw:: html

   <div style="margin-top: 1em"></div>

.. [17] Fourier Neural Operator based surrogates for CO2 storage in realistic geologies,
        Anirban Chandra, Marius Koch, Suraj Pawar, Aniruddha Panda, 
        Kamyar Azizzadenesheli, Jeroen Snippe, Faruk O Alpak, Farah Hariri, 
        Clement Etienam, Pandu Devarakota, Anima Anandkumar, Detlef Hohl, 2025.
        arXiv:2503.11031.

.. raw:: html

   <div style="margin-top: 1em"></div>

.. [18] Huge ensembles part i: Design of ensemble weather forecasts using spherical Fourier neural operators,
        Ankur Mahesh, William Collins, Boris Bonev, Noah Brenowitz, Yair Cohen, 
        Joshua Elms, Peter Harrington, Karthik Kashinath, Thorsten Kurth, 
        Joshua North, 2024.
        arXiv:2408.03100.

.. raw:: html

   <div style="margin-top: 1em"></div>

.. [19] Spherical Fourier neural operators: learning stable dynamics on the sphere,
        Boris Bonev, Thorsten Kurth, Christian Hundt, Jaideep Pathak, Maximilian Baust, 
        Karthik Kashinath, Anima Anandkumar, 2023.
        Proceedings of the 40th International Conference on Machine Learning (ICML).

.. raw:: html

   <div style="margin-top: 1em"></div>

.. [20] Fourier Neural Operators for Arbitrary Resolution Climate Data Downscaling,
        Qidong Yang, Paula Harder, Venkatesh Ramesh, Alex Hernandez-Garcia, 
        Daniela Szwarcman, Prasanna Sattigeri, Campbell D Watson, David Rolnick, 2023.
        ICLR 2023 Workshop on Tackling Climate Change with Machine Learning.

.. raw:: html

   <div style="margin-top: 1em"></div>

.. [21] Tipping Point Forecasting in Non-Stationary Dynamics on Function Spaces,
        Miguel Liu-Schiaffini, Clare E Singer, Nikola Kovachki, Tapio Schneider, 
        Kamyar Azizzadenesheli, Anima Anandkumar, 2023.
        arXiv:2308.08794.

.. raw:: html

   <div style="margin-top: 1em"></div>

.. [22] Ultrasound Lung Aeration Map via Physics-Aware Neural Operators,
        Jiayun Wang, Oleksii Ostras, Masashi Sode, Bahareh Tolooshams, Zongyi Li, 
        Kamyar Azizzadenesheli, Gianmarco Pinton, Anima Anandkumar, 2025.
        arXiv:2501.01157.

.. raw:: html

   <div style="margin-top: 1em"></div>

.. [23] A Unified Model for Compressed Sensing MRI Across Undersampling Patterns,
        Armeet Singh Jatyani, Jiayun Wang, Aditi Chandrashekar, Zihui Wu, 
        Miguel Liu-Schiaffini, Bahareh Tolooshams, Anima Anandkumar, 2024.
        arXiv:2410.16290.

.. raw:: html

   <div style="margin-top: 1em"></div>

.. [24] AI-aided geometric design of anti-infection catheters,
        Tingtao Zhou, Xuan Wan, Daniel Zhengyu Huang, Zongyi Li, Zhiwei Peng, 
        Anima Anandkumar, John F Brady, Paul W Sternberg, Chiara Daraio, 2024.
        Science Advances, 10(1).

.. raw:: html

   <div style="margin-top: 1em"></div>

.. [25] Neural Operator Learning for Ultrasound Tomography Inversion,
        Haocheng Dai, Michael Penwarden, Robert M Kirby, Sarang Joshi, 2023.
        arXiv:2304.03297.

.. raw:: html

   <div style="margin-top: 1em"></div>

.. [26] Neural Born Series Operator for Biomedical Ultrasound Computed Tomography,
        Zhijun Zeng, Yihang Zheng, Youjia Zheng, Yubing Li, Zuoqiang Shi, He Sun, 2023.
        arXiv:2312.15575.

.. raw:: html

   <div style="margin-top: 1em"></div>

.. [27] Graph Neural Operators for Classification of Spatial Transcriptomics Data,
        Junaid Ahmed, Alhassan S Yasin, 2023.
        arXiv:2302.00658.

.. raw:: html

   <div style="margin-top: 1em"></div>

.. [28] Adaptive Fourier neural operators: Efficient token mixers for transformers,
        John Guibas, Morteza Mardani, Zongyi Li, Andrew Tao, Anima Anandkumar, 
        Bryan Catanzaro, 2021.
        arXiv:2111.13587.

.. raw:: html

   <div style="margin-top: 1em"></div>

.. [29] Fast sampling of diffusion models via operator learning,
        Hongkai Zheng, Weili Nie, Arash Vahdat, Kamyar Azizzadenesheli, Anima Anandkumar, 2023.
        International Conference on Machine Learning.

.. raw:: html

   <div style="margin-top: 1em"></div>

.. [30] Resolution-invariant image classification based on Fourier neural operators,
        Samira Kabri, Tim Roith, Daniel Tenbrinck, Martin Burger, 2023.
        International Conference on Scale Space and Variational Methods in Computer Vision.

.. raw:: html

   <div style="margin-top: 1em"></div>

.. [31] FNOSeg3D: Resolution-Robust 3D Image Segmentation with Fourier Neural Operator,
        Ken CL Wong, Hongzhi Wang, Tanveer Syeda-Mahmood, 2023.
        2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI).


.. raw:: html

   <div style="margin-top: 3em;"></div>