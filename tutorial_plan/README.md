discuss part:
1. hierarchy of beginner, advanced, FAQ
2. scheduler in beginner or advanced


# tutorial general structure
    * overview
    * (content)
    * related Apphub (optional)


# tutorial:

## beginner

### t01_getting_started - YC
* Pipeline *> Network *> Estimator
* Simple example

### t02_dataset - PB
* pytorch dataset flashback (our dataset is based on their dataset class)
            https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    * dataset[0]
    * len(dataset)

* FE dataset
    * Data from disk
        * LabeledDirDataset
            * prepare files
                /some/temp/folder/:
                    /a
                        a1.txt
                        a2.txt
                    /b
                        b1.txt
                        b2.txt

            call ds[0]

        * CSV dataset
            data    label
             a1.txt      0
             a2.txt      0
                ..       1
                ..       1

        len(ds) = 4,
        ds[1] = {"x": 'a1.txt', 'y':0}

    * Data from memory
        * NumpyDataset:


## t03_Operator - XD
* Operator Concept
* inputs/outputs/
* mode
* Operator connections
* examples


## t04_Pipeline - VS
* DL needs preprocessing - > pipeline -> CPU

* Pipeline API
   * train/eval/test:
        pipeline_tf * tf.data.Dataset (example) -
        pipeline_torch * pytorch dataloader (example) -
        pipeline_fe * import from existing FE dataset /simple torch dataset -

   * pipeline.get_results(batch_size:4)
     call all of instances

* NumpyOp - if you are using FEdataset/torchdataset
    * Concept (Non-tensor) (inheriting Op - reference tutotial 3) - lots of augmentation from albumentajtion

    * univarite (same forward regardless of feature)
        multi-io:  Minmax(inputs=("x", "y"), out=("x", "y"))
    * multivariate (different forward dependent on feature )
        bbox,image.  use one example

    * customizing numpyOp:
        *(add random noise)

* Example:
    mnist/cifar:
        Normalize(univariate)
        Rotation(multi-variate)
        Add random noise
        use them in Pipeline
        call get_results
        visualize the results



## t05_Model - GK
* Define model function(pytorch, tensorflow)
    * import FE architecture
    * import tf.keras.application
    * import torchvision pretrained model
* Compile model
    * fe.build
    * Define optimizer
        * string
        * lambda function
    * load model weight
    * model name

## t06_Network -YC
* TensorOp
    * ModelOp
    * UpdateOp
* CustomizeTensorOp(simple example)
    * tf
    * torch
    * backend


## t07_Estimator - XD
    * Estimator
        * change logging behavior
        * max_steps_per_epoch
        * monitor_names

    * Trigger fit, test


    * Trace:
        * concept
        * structure
        * example(model_saver)



# t08 mode - XD



# t09 Inference - YC
* data[0]
* pipeline transform
* network transform

* t10 Cli usage - PB
        * mechanism(cli: looking for get_estimator,  **arg is replacing the arg in get_estimator)
        * fastestiamtor train
        * fastestiamtor test
        * fastestiamtor train --hyperparameters param.json




## advanced

* t01. FE dataset - advanced
        * how to customize dataset (point to pytorch tutorial
        * how to split
            * based on sample index
            * probability
            * number
            * muti*split (multiple of above)
        * summary
            * ds.summary
            * print(ds)
        * global dataset editing:
            * apply sk.standardize (apphub tabular) ?
        * how to create dataset from disk
            * LabeledDirDataset
            * CSV dataset
            * generator_dataset
            * DirDataset * (unlabeledDataset)
            * LabeledDirDataset
            * generator_dataset
        * batch dataset
            * deterministic batching: batch=8, 5 postive, 3 negative (same keys)
            * stochastic batching: batch=8, 0.8 from postive, 0.2 from negative
            * unpaired dataset batch=2, one from horse, one from zebra (different keys)


* t02 pipeline-advanced
        * pipeline padding
        * get_loader: (loop through dataset)
        * benchmark

* t03   op-advanced:
        * op mechanism
            * state
            * data (how to grab data)
        * NumpyOp:
            * deleteOp
            * Customize numpyOp
            * meta

        *TensorOp:
            * why we don't need `DeleteOp` * we already have gpu key filtering in network
           * customize tensorOp
           * backend

 * t04   Trace-advanced
        * How multiple metrics work together
            * ( T1: accuracy, T2: f1_score, )
        * How to customize Trace:
            * for debugging, monitoring/ calculating metrics
            * trace time point (epoch_start, batch_start ....)
        * Demo saving models


* t05 Scheduler:
        * Scheduler Basics: (epoch is unit time for scheduler)
            * EpochScheduler: {epoch: content} : {1: x, 3: None 4: y}:
                *number means epoch of change.
                * use None as op if no op
            * RepeatScheduler: [list of content] one example: intermitent adversarial training
        * things you can schedule:
            * Pipeline:
                * schedule data: train_data, eval_data: (transfer learning)
                * batch_size
                * ops
            * Network:
                * ops
                * optimizer


* t06 Summary
       * Accessing history in python way
            summary = est.fit(summary="experiment1")
            summary = est.test(summary="experiment1")

       * TensorBoard

       * log parsing &experiment logging (visualization):
            * use summary object
            * use trainig log txt

* t07 learning rate scheduling - GK
    * provide lambda function, then use 'epoch' or 'step' as argument name- check cifar10
        * epoch
        * step
    * use exsiting lr shceudler(cosine/cyclic coscine/ linear decay) - check mnist

* t08
     XAI
        * Saliency tutorial