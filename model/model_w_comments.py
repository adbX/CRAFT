import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import os

from model.config import *
from model.data import *
from model.model import *

def maskNLLLoss(inp, target, mask):
    """
    Compute the Negative Log-Likelihood loss, but only for the non-padded tokens
    (determined by `mask`). This helps ignore loss contributions from padding.
    """
    # Calculate the total number of valid tokens in the batch
    nTotal = mask.sum()
    # Gather the predicted probabilities for the correct target tokens
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    # Compute mean cross-entropy over only non-padding tokens
    loss = crossEntropy.masked_select(mask).mean()
    # Move the final loss value to the device
    loss = loss.to(device)
    # Return the mean loss and the total number of valid tokens
    return loss, nTotal.item()

def train(input_variable, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices, 
          target_variable, mask, max_target_len,                                                           
          encoder, context_encoder, decoder, embedding,                                                    
          encoder_optimizer, context_encoder_optimizer, decoder_optimizer,                                 
          batch_size, clip, max_length=MAX_LENGTH):
    """
    Performs a single training iteration (forward pass + backward pass) on one batch.
    """
    # Reset (zero) the gradients in the optimizers
    encoder_optimizer.zero_grad()
    context_encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Move all relevant tensors to the current device (CPU or GPU)
    input_variable = input_variable.to(device)
    dialog_lengths = dialog_lengths.to(device)
    utt_lengths = utt_lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables for the loss calculation
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through the utterance encoder
    #  - input_variable: a padded batch of token IDs for each utterance
    #  - utt_encoder_hidden: the final hidden states of the utterance encoder
    _, utt_encoder_hidden = encoder(input_variable, utt_lengths)
    
    # Reshape/reorder the final utterance encoder states so that they form
    # a sequence of utterance vectors for each entire dialog
    context_encoder_input = makeContextEncoderInput(
        utt_encoder_hidden,
        dialog_lengths_list,
        batch_size,
        batch_indices,
        dialog_indices
    )
    
    # Forward pass through the context encoder
    #  - context_encoder_input: the sequence of utterances for each dialog
    #  - dialog_lengths: lengths of the sequences (number of utterances per dialog)
    context_encoder_outputs, context_encoder_hidden = context_encoder(context_encoder_input, dialog_lengths)

    # Create the initial decoder input, which is a batch of SOS tokens
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Initialize the decoder's hidden state with the final hidden state of the context encoder
    # (The decoder may have multiple layers, so we take as many hidden states as the decoder has layers.)
    decoder_hidden = context_encoder_hidden[:decoder.n_layers]

    # Decide whether to use teacher forcing for this batch (compare random number to ratio)
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward pass through the decoder, one token at a time
    if use_teacher_forcing:
        # If teacher forcing: feed the ground truth token at each step
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, 
                decoder_hidden, 
                context_encoder_outputs
            )
            # Teacher forcing: next input is the current ground-truth token
            decoder_input = target_variable[t].view(1, -1)
            # Compute the masked NLL loss for the current step
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            # Accumulate total loss
            loss += mask_loss
            # Keep track of loss for printing
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        # If not using teacher forcing: use the model's own predicted token at each step
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, 
                decoder_hidden, 
                context_encoder_outputs
            )
            # Select the top predicted token
            _, topi = decoder_output.topk(1)
            # Reshape that predicted token for the next decoder input
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Compute the masked NLL loss at this step
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            # Accumulate the loss
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Backpropagate the accumulated loss
    loss.backward()

    # Clip gradients to mitigate exploding gradients
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(context_encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Update the model parameters with the gradients
    encoder_optimizer.step()
    context_encoder_optimizer.step()
    decoder_optimizer.step()

    # Return the average loss for this batch
    return sum(print_losses) / n_totals

def trainIters(voc, pairs, encoder, context_encoder, decoder,
               encoder_optimizer, context_encoder_optimizer, decoder_optimizer, embedding, 
               encoder_n_layers, context_encoder_n_layers, decoder_n_layers, 
               save_dir, n_iteration, batch_size, print_every, clip, corpus_name):
    """
    High-level function that handles the training loop across many iterations.
    """
    # Create a batch iterator for the training data
    batch_iterator = batchIterator(voc, pairs, batch_size)
    
    # Initialization for the training loop
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0

    # Begin the training process
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        # Retrieve the next batch of dialogs and related data
        training_batch, training_dialogs, _, true_batch_size = next(batch_iterator)
        # Extract fields from the batch
        input_variable, dialog_lengths, utt_lengths, batch_indices, dialog_indices, labels, _, target_variable, mask, max_target_len = training_batch
        # A list of how many utterances are in each dialog of this batch
        dialog_lengths_list = [len(x) for x in training_dialogs]

        # Run one training iteration on the current batch
        loss = train(
            input_variable, 
            dialog_lengths, 
            dialog_lengths_list, 
            utt_lengths, 
            batch_indices, 
            dialog_indices,
            target_variable, 
            mask, 
            max_target_len,
            encoder, 
            context_encoder, 
            decoder, 
            embedding,
            encoder_optimizer, 
            context_encoder_optimizer, 
            decoder_optimizer,
            true_batch_size, 
            clip
        )
        # Accumulate the loss for averaging
        print_loss += loss
        
        # Print the average loss at regular intervals (print_every)
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(
                iteration, 
                iteration / n_iteration * 100, 
                print_loss_avg
            ))
            # Reset the accumulated loss
            print_loss = 0

if __name__ == "__main__":
    # Set the random seed for reproducibility
    random.seed(2019)

    # Detect whether a GPU (CUDA) is available and set the device accordingly
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    print("Using device", ("cuda" if USE_CUDA else "cpu"))

    # Load the precomputed vocabulary
    voc = loadPrecomputedVoc(corpus_name, word2index_path, index2word_path)
    # Load unlabeled training data (dialog pairs)
    train_pairs = loadUnlabeledData(voc, train_path)

    print('Building encoders and decoder ...')
    # Create the word embedding layer
    embedding = nn.Embedding(voc.num_words, hidden_size)
    # Initialize the encoder, context encoder, and decoder with the given configuration
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    context_encoder = ContextEncoderRNN(hidden_size, context_encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

    # Move models to the chosen device
    encoder = encoder.to(device)
    context_encoder = context_encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')

    # Compute total training iterations based on the number of epochs and the size of the dataset
    n_iter_per_epoch = len(train_pairs) // batch_size + int(len(train_pairs) % batch_size == 1)
    n_iteration = n_iter_per_epoch * pretrain_epochs

    # Ensure dropout and other layers are in training mode
    encoder.train()
    context_encoder.train()
    decoder.train()

    # Initialize the Adam optimizers for each component
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    context_encoder_optimizer = optim.Adam(context_encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

    # Start the training loop
    print("Starting Training!")
    print("Will train for {} iterations".format(n_iteration))
    trainIters(
        voc, 
        train_pairs, 
        encoder, 
        context_encoder, 
        decoder,
        encoder_optimizer, 
        context_encoder_optimizer, 
        decoder_optimizer, 
        embedding, 
        encoder_n_layers, 
        context_encoder_n_layers, 
        decoder_n_layers, 
        save_dir, 
        n_iteration, 
        batch_size,
        print_every, 
        clip, 
        corpus_name
    )

    # Save the trained model parameters
    print("Saving!")
    directory = os.path.join(save_dir, corpus_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save({
        'en': encoder.state_dict(),
        'ctx': context_encoder.state_dict(),
        'de': decoder.state_dict(),
        'en_opt': encoder_optimizer.state_dict(),
        'ctx_opt': context_encoder_optimizer.state_dict(),
        'de_opt': decoder_optimizer.state_dict(),
        'voc_dict': voc.__dict__,
        'embedding': embedding.state_dict()
    }, os.path.join(directory, "model.tar"))
