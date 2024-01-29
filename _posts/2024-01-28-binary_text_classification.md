---
layout: post
category: Machine Learning
tag: Natural Language Processing
---

# Binary Text Classification using PyTorch and HuggingFace


---
_table of contents_
1. TOC
{:toc}
---

## Before Beginning

This project was written months ago when I was doing my intern at a conversational AI company. They wanted to test their in-house purified data with well-known transformers. Since then, I deleted all private credentials and change the dataset to [IMDB Large Movie Review][dataset]. Feel free to try another binary text classification datasets.

Unfortunately, I changed some portions of the code, too. Especially, data preparation parts. So If you see some leftover code, just ignore them.

## General Concepts

Every AI project begins with gathering data. There are numerous methods available from web-scraping, APIs, crowdsourcing, simulation data and, of course, public datasets. This part will bother you with ethical considerations and legal implications.

Our main aim is to separate positive and negative comments on the films. There are other variants such as regression of stars the user give to a product, or in our case the film. This is another topic but probably these models can be used here as well. If the model is really good, or you just want to test it out such models just change the last layers for binary classicifation. Additionally, it is advised to run these network while the lower layers are freezed for a few epochs and then unfreezing all the layers to continue to train. This way, the model parameters will not be changed too much, as they are already trained and learned something. Just remember to freeze the lower layers for a few epochs.

We have already configured binary classification models using HuggingFace. They provide very easy to use functions such as `AutoModelForSequenceClassification`.

I also designed a few things to give feedback to the user using some progress bar thanks to `tqdm`. TensorFlow had such things and I didn't see anybody implement this.

The manual_classification.py and trainer_classification.py are the main .py files that trains the models. Other files are supporters and some other topics which will be explained later.

---
NOTE

Is transfer learning cheating?

This is a funny question, but it used to be a question I thought in the past. Well, of course not. After all, in engineering, you are trying to use resources correctly and find the best solution. On the other hand, wanting to design these things one day isn't bad, is it? Isn't it?

---


## File Navigation

Let's see which file does what.

There are two .py files to train and test the models.

- manual_classification.py --> Fetch the model, preprocess the dataset and convert it to HuggingFace Dataset format, split the dataset into validation and train the model


- trainer_classification.py --> Same as above, but this approach using Trainer of HuggingFace. Avoid this approach, as you cannot control the things inside it.


- configs.py --> global variables used in various scripts. User can change these variables for their own ease. Seeing all variables in one place makes adjustments very easy and effective.


- utility_functions.py --> These functions designed to re-usable. They abstract many things and packed them into several functions.


- split_csv_script.py --> As I have converted my old codes, I needed to add this script to split the dataset into train and test datasets. The train dataset will be split later on the *_classification.py files.


- intel_compressor.py --> Intel Neural Compressor is used for quantization, pruning, and knowledge distillation for various frameworks [1]. I just added these two while there were nothing to do in the job. They can be useful if you are using a 10-year-olds computer like me or you want to deploy your model to edge.


- intel_compressor_test.py --> It simply tries the compressed model.

## Working Principle

### The Main Script

I highly encourage you to take a look at my source. I wrote too many documentations and I don't want to steal your time further. Check out my GitHub. Just a few notes are shared here.

Let's see the entire manual_classification.py. 

	{% highlight python %}
if __name__ == '__main__':
    torch.cuda.empty_cache()

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=training_args['num_labels'], ignore_mismatched_sizes=True).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- dataset settings ---
    train_set_raw, valid_set_raw = get_raw_csv(training_args['train_filepath'],
                                               reduce_ratio=training_args['reduce_ratio'],
                                               valid_size=training_args['valid_size'],
                                               return_valid=training_args['return_valid'])
    y_train = train_set_raw['label']
    y_valid = valid_set_raw['label']
    train_negative_ratio = y_train[y_train == 0].count() / y_train[y_train == 1].count()
    print_dataset_info(y_train, y_valid)

    train_set = process_dataset(train_set_raw, tokenizer, tokenizer_config)
    valid_set = process_dataset(valid_set_raw, tokenizer, tokenizer_config)
    # --------------------------

    # --- training arguments ---
    # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    criterion = nn.CrossEntropyLoss() if training_args['num_labels'] > 2 else nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([train_negative_ratio], device=device))

    # optimizer = torch.optim.NAdam(model.parameters(), lr=1e-4, )
    optimizer = training_args['optimizer'](params=model.parameters())
    scheduler = training_args['scheduler'](optimizer=optimizer)
    accuracy_metric = training_args['accuracy_metric']
    f1_metric = training_args['f1_metric']  # for such an imbalanced dataset, accuracy is not a good metric

    # --------------------------

    # print('model summary:', model)

    data_collator = training_args['data_collator'](tokenizer=tokenizer)

    train_loader = DataLoader(train_set, batch_size=training_args['batch_size'], drop_last=True,
                              collate_fn=data_collator)
    valid_loader = DataLoader(valid_set, batch_size=training_args['batch_size'], drop_last=True,
                              collate_fn=data_collator)

    train(model, train_loader, valid_loader, optimizer, scheduler, criterion, path_to_save)

    # load the best model in terms of specific metric, this one uses f1_score for comparisons
    model.load_state_dict(torch.load(path_to_save))
    print('\n[INFO] The best model is loaded\n')

    # --- lastly, load the test dataset and print some predictions ---
    predict_test_set(model, data_collator, tokenizer, criterion)
   {% endhighlight %}

The main steps are given as:

1. Fetch the pretrained model. These models are fetched from [HuggingFace](https://huggingface.co/). They have all sorts of models for nearly every job. I choose 3 models that works directly with our implementation.

2. Split the dataset into train and validation datasets. As you will see later, the train f1 scores are 0.91 but the validation scores remains at 0.803. We need to see how well our model is during after each epoch.

3. Process the dataset. If needed, feature extraction can be made. But our dataset is relatively easy and we will not going into detail.

4. Tokenize the dataset. HuggingFace models have their own tokenizers, and you should be using these as the model recognize those tokens.

5. Set training environment. optimizer, scheduler, and metrics are chosen

6. Train the model. This is simply a for loop and backpropagation.

7. Load the best model w.r.t. your validation metric. It was f1 metric in our case and after each epoch the model is saved based on its score.

8. Predict on the test set. The score of test set is the final value we are looking for. Validation and train scores are just indicators. This is why we are doing too much work.


### Utility Functions

There are several utility functions to abstract the processes from the main file. Someone can also write unit tests for these, for which I don't have time.

For example, the train_one_epoch() is called by train() for each epoch. Sidenote: we are using tensorboard to track the training data. Explained further in configs.py

    {% highlight python %}

def train_one_epoch(model, train_loader, optimizer, scheduler, criterion, epoch_index):
    """
    Train the model for one epoch using the provided training data.

    Args:
        model: The PyTorch model to be trained.
        train_loader: DataLoader containing training data.
        optimizer: The optimizer used for training.
        scheduler: Learning rate scheduler.
        criterion: The loss criterion used for optimization.
        epoch_index (int): The index of the current epoch.

    Returns:
         None
    """
    calculate_loss = calculate_loss_for_multi if training_args['num_labels'] > 2 else calculate_loss_for_binary
    dataloader_prog = tqdm.tqdm(train_loader, position=0, leave=True)
    model.train()

    running_loss = 0.0
    for i, batch in enumerate(dataloader_prog):
        input_ids, mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(
            device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=mask).logits
        loss: torch.Tensor = calculate_loss(labels, outputs, criterion)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        dataloader_prog.set_description(
            f'epoch: {epoch_index + 1} -> loss: {loss.item():.3e}, lr: {scheduler.get_last_lr()[0]:.3e}, '
            f'gpu_util: %{get_gpu_util()} ::')

        predicted_labels = torch.argmax(outputs, dim=1)
        training_args['accuracy_metric'].add_batch(predictions=predicted_labels, references=labels)
        training_args['f1_metric'].add_batch(predictions=predicted_labels, references=labels)

    scheduler.step()
    epoch_loss = running_loss / len(train_loader)  # summed losses during epoch / number of batches
    train_acc = training_args["accuracy_metric"].compute()["accuracy"]
    train_f1 = training_args["f1_metric"].compute(average="macro")["f1"]
    print(
        f'epoch: {epoch_index + 1} -> '
        f'train_loss: {epoch_loss: .3e}, '
        f'train_acc: {train_acc: .3f}, '
        f'train_f1: {train_f1: .3f}')

    if training_args['writer'] is not None:
        training_args['writer'].add_scalar('loss/train', epoch_loss, global_step=epoch_index + 1)
        training_args['writer'].add_scalar('lr/train', scheduler.get_last_lr()[0], global_step=epoch_index + 1)
        training_args['writer'].add_scalar('gpu_util/train', get_gpu_util(), global_step=epoch_index + 1)
        training_args['writer'].add_scalar('train_acc/train', train_acc, global_step=epoch_index + 1)
        training_args['writer'].add_scalar('train_f1/train', train_f1, global_step=epoch_index + 1)


   {% endhighlight %}




### Configs

Configs file is usually working with dictionaries, which have various global parameters to be used inside utility functions. Collecting them in one place makes the project much more organized and well-structured and it helps to see all hyperparameters in one place.



Example is given below. Maybe it can be organized much efficiently, but this is working, too.


    {% highlight c %}

training_args = dict(
    path_to_save=Path(fr'checkpoints/{saving_name}.pt'),
    epochs=10,
    batch_size=96,
    num_labels=2,  # try out num_labels=2, automatically chooses negative and positive ones.
    optimizer=partial(torch.optim.NAdam, lr=1e-5),
    # optimizer=partial(torch.optim.AdamW, lr=1e-5),
    # choosing criterion in the main script would be better for code readability
    scheduler=partial(torch.optim.lr_scheduler.StepLR, step_size=2, gamma=0.95),
    accuracy_metric=evaluate.load('accuracy'),
    f1_metric=evaluate.load("f1", average='macro'),
    # for such an imbalanced dataset, accuracy is not a good metric
    # average='macro' probably not working and throws no error, too! Implement so f1_metric.compute(.., average='macro')
    data_collator=DataCollatorWithPadding,
    train_filepath=r'datasets/imdb_binary/test.csv',
    test_filepath=r'datasets/imdb_binary/train.csv',
    sentiment_mapping={'positive': 1, 'negative': 0},
    reduce_ratio=None,  # use only %reduce_ratio of the dataset, 0.1 -> %10 of train_set
    valid_size=0.2,
    return_valid=True,
    writer=SummaryWriter()  # turn off tensorboard by setting -> writer=None
    # how to see tensorboard results -> open terminal in virtual env -> tensorboard --logdir=runs
)

   {% endhighlight %}

## Conclusion

This project was a demonstration of using PyTorch along Huggingface and various other python packages. As a ex Tensorflow user, I would admit that the main idea behind the machine learning is obviously same and I got running this scripts in just a few weeks. PyTorch seems to be a very good library. Bye.

[dataset]: http://ai.stanford.edu/~amaas/data/sentiment/
[1]: https://www.intel.com/content/www/us/en/developer/tools/oneapi/neural-compressor.html#gs.3lcn5q