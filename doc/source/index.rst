:no-toc:
:no-localtoc:
:no-pagination:

.. neural-operator documentation

.. only:: html

   .. raw:: html

      <br/><br/>

.. only:: html

   .. raw:: html 
   
      <div class="has-text-centered">
         <h2> Neural Operators in PyTorch </h2>
      </div>
      <br/><br/>

.. only:: latex

   Neural Operators in PyTorch
   ===========================

.. image:: _static/logos/neuraloperator_logo_long.png
   :align: center
   :width: 500


``neuraloperator`` is a comprehensive library for 
learning neural operators in PyTorch.
It is the official implementation for Fourier Neural Operators 
and Tensorized Neural Operators.

Unlike regular neural networks, neural operators
enable learning mapping between function spaces, and this library
provides all of the tools to do so on your own data.

NeuralOperators are also resolution invariant, 
so your trained operator can be applied on data of any resolution.

Quickstart
==========

This guide will walk you through the standard ML workflow of loading data, creating a neural operator, training it on your data and saving the trained model for later use.

First install the library ``pip install neuraloperator`` (see :doc:`install` for more options).

To create a Fourier Neural Operator model:


.. code-block:: python

   from neuralop.models import FNO

   operator = FNO(n_modes=(16, 16), hidden_channels=64,
                   in_channels=3, out_channels=1)

To save the weights of the trained model:

.. code-block:: python

   model.save_checkpoint(save_folder='./checkpoints/', save_name='example_fno')

And to load the weights later:

.. code-block:: python
   
   from neuralop.models import FNO
   model = FNO.from_checkpoint(save_folder='./checkpoints/', save_name='example_fno')

``neuraloperator`` comes prepackaged with an example dataset of flows governed by the Darcy flow equation. 

To import the data:

.. code-block:: python

   import torch
   from neuralop.datasets import load_darcy_flow_small

   train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=1000, batch_size=32, 
        test_resolutions=[32], n_tests=[100],
        test_batch_sizes=[32],
        positional_encoding=True
)

Similar to the API provided by ``torchvision``, this dataset includes training and test data for use in standard PyTorch training loops,
as well as a ``preprocessor`` object that automates the transforms to convert the data into the form best understood by the model. 

We provide a ``Trainer`` object that automates the logic of a basic neural operator training loop to speed up experimentation (see :doc: `auto_examples` for more information).

.. code-block:: python

   from neuralop.training import Trainer

   # Create the trainer
   trainer = Trainer(model=model, n_epochs=20,
                     data_processor=data_processor,
                     wandb_log=False,
                     log_test_interval=3,
                     use_distributed=False,
                     verbose=True)

   # train the model
   trainer.train(train_loader=train_loader,
              test_loaders=test_loaders,
              optimizer=optimizer,
              scheduler=scheduler, 
              regularizer=False, 
              training_loss=train_loss,
              eval_losses=eval_losses)

Tensorization is also provided out of the box: you can improve the previous models
by simply using a Tucker Tensorized FNO with just a few parameters:

.. code-block:: python

   from neuralop.models import TFNO

   operator = TFNO(n_modes=(16, 16), hidden_channels=64,
                   in_channels=3, 
                   out_channels=1,
                   factorization='tucker',
                   implementation='factorized'
                   rank=0.05)

This will use a Tucker factorization of the weights. The forward pass
will be efficient by contracting directly the inputs with the factors
of the decomposition. The Fourier layers will have 5% of the parameters
of an equivalent, dense Fourier Neural Operator!


.. toctree::
   :maxdepth: 1
   :hidden:

   install
   user_guide/index
   modules/api
   auto_examples/index


.. only:: html

   .. raw:: html

      <br/> <br/>
      <br/>

      <div class="container has-text-centered">
      <a class="button is-medium is-dark is-primary" href="install.html">
         Install
      </a>
      </div>


      <!-- CITE -->
      <div class="container mt-6 pt-6">
         <div class="card">
         <div class="card-content">
         <p>
            If you use NeuralOperator, please cite the following papers:
         </p>
         <p>
            <it> Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., and Anandkumar A. </it>, 
            <strong> “Fourier Neural Operator for Parametric Partial Differential Equations”</strong>, 
            ICLR, 2021. 
            <br/> <a href="https://arxiv.org/abs/2010.08895">https://arxiv.org/abs/2010.08895</a>.
         </p>
         <p>
            <it> Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A., and Anandkumar A. </it>,
            <strong>  “Neural Operator: Learning Maps Between Function Spaces”, </strong>, 
            JMLR, 2021. 
            <br/> <a href="https://arxiv.org/abs/2108.08481">https://arxiv.org/abs/2108.08481</a>.
         </p>

         <blockquote id="bibtex" class="is-hidden">
            @misc{li2020fourier,<br/>
            &emsp; title={Fourier Neural Operator for Parametric Partial Differential Equations}, <br/>
            &emsp; author={Zongyi Li and Nikola Kovachki and Kamyar Azizzadenesheli and Burigede Liu and Kaushik Bhattacharya and Andrew Stuart and Anima Anandkumar},<br/>
            &emsp; year={2020},<br/>
            &emsp; eprint={2010.08895},<br/>
            &emsp; archivePrefix={arXiv},<br/>
            &emsp; primaryClass={cs.LG}<br/>
            } <br/> <br/>
            @article{kovachki2021neural,<br/>
            &emsp;  author    = {Nikola B. Kovachki and Zongyi Li and Burigede Liu and Kamyar Azizzadenesheli and Kaushik Bhattacharya and Andrew M. Stuart and Anima Anandkumar},<br/>
            &emsp;  title     = {Neural Operator: Learning Maps Between Function Spaces},<br/>
            &emsp;  journal   = {CoRR},<br/>
            &emsp;  volume    = {abs/2108.08481},<br/>
            &emsp;  year      = {2021},<br/>
            }<br/>
            <br/>
         </blockquote>
         </div>
   
         <footer class="card-footer">
         <p class="card-footer-item">
         <a onclick="javascrip:toggle_bibtex();" >
            <span class="button" id="bibtex-toggle">show bibtex</span>
         </a>
         </p>
         </footer>
   
         </div>
      </div>
   
      <script>
         function toggle_bibtex() {
            var bibtex = document.getElementById("bibtex");
            var toggle = document.getElementById("bibtex-toggle");
            bibtex.classList.toggle('is-hidden');
            if (toggle.textContent == 'show bibtex') {
               toggle.textContent = 'hide bibtex';
            }
            else {
               toggle.textContent = 'show bibtex';
            }
         };
      </script>