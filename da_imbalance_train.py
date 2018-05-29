__author__ = 'xiangyang'

import sys
import tensorflow as tf
import numpy as np
import pickle
import os
import random
from da_model import LSTMPrediction
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, auc, roc_curve

tf.app.flags.DEFINE_integer("num_classes", 2, "number of labels")
tf.app.flags.DEFINE_float("learning_rate", 0.005, "learning rate of algorithm")
tf.app.flags.DEFINE_integer("batch_size", 32, "batch size for training/evaluating")
tf.app.flags.DEFINE_integer("decay_steps", 10000, "How many steps before decay learning rate")
tf.app.flags.DEFINE_float("decay_rate", 0.9, "Speed of decay for learning rate")
tf.app.flags.DEFINE_string("ckpt_dir", "./checkpoint/", "directory for storing checkpoint of the model")  #
tf.app.flags.DEFINE_integer("sequence_length", 128, "maximum visit length")
tf.app.flags.DEFINE_integer("icd_length", 32, "maximum icd length")
tf.app.flags.DEFINE_integer("med_length", 32, "maximum med length")
tf.app.flags.DEFINE_integer("external_feature_len", 3, "external feature length")
tf.app.flags.DEFINE_integer("icd_embed_size", 100, "diag embedding dimension")
tf.app.flags.DEFINE_integer("med_embed_size", 100, "med embedding dimension")
tf.app.flags.DEFINE_integer("visit_encode_size", 100, "diag embedding dimension")
tf.app.flags.DEFINE_integer("hidden_size", 100, "hidden layer dimension")
tf.app.flags.DEFINE_boolean("is_training", True, "distinguish train and test")
tf.app.flags.DEFINE_integer("num_epochs", 10000, "number of training iterations")
tf.app.flags.DEFINE_integer("validate_every", 100, "validate in every epoch(s)")
tf.app.flags.DEFINE_boolean("use_embedding", True, "whether to use pretrained embedding or not")
tf.app.flags.DEFINE_string("training_path", "./data/dataset.pkl", "path of training data")
tf.app.flags.DEFINE_string("dict_path", "./data/dict.pkl", "path of training data")
tf.app.flags.DEFINE_string("prediction_path", "./evaluation/", "directory to store predictions")
FLAGS = tf.app.flags.FLAGS
random.seed(2018)


# prepare a sentence
def make_sent(words, word2id, max_len):
    x = []
    for word in words:
        if word in word2id:
            x.append(word2id[word])
        else:
            x.append(0)
    m = max_len - len(x)
    for i in range(0, m):
        x.append(0)
    return x


# prepare a document for the model
def make_doc(icds, meds, patient_time_infos, patient_ext, icd2id, med2id, max_visit_len, max_icd_len, max_med_len):
    d = []
    # for each icd or med code, the last is 3 time features for each visit
    NIL_sentence = np.zeros((max_icd_len + max_med_len + 3))
    for i in range(len(icds)):
        icd = make_sent(icds[i], icd2id, max_icd_len)
        med = make_sent(meds[i], med2id, max_med_len)
        d.append(np.concatenate([icd, med, patient_time_infos[i]]))  # a sentence
    m = max_visit_len - len(d)
    for i in range(0, m):
        d.append(NIL_sentence)
    # external features
    d.append(patient_ext)
    return d


def prepare_test_data(labels, icd_codes, med_codes, times, exts, icd2id, med2id,
                      max_visit_len, max_icd_len, max_med_len):
    X, Y = [], []
    # print (times)
    # traverse each patient
    for i in range(len(labels)):
        label = labels[i]
        icds = icd_codes[i]
        meds = med_codes[i]
        patient_time_infos = times[i]  # three time features for each visit
        patient_ext = exts[i]  # three demographic features for each patient
        flag = 0
        if len(icds) > max_visit_len or len(meds) > max_visit_len or len(patient_time_infos) > max_visit_len:
            flag = 1
        else:
            for sentence in icds:
                if len(sentence) > max_icd_len:
                    flag = 1
                    break
            for sentence in meds:
                if len(sentence) > max_med_len:
                    flag = 1
                    break
        # over the maximum length, then pass
        if flag == 1:
            continue
        doc = make_doc(icds, meds, patient_time_infos, patient_ext,
                       icd2id, med2id, max_visit_len, max_icd_len, max_med_len)
        X.append(doc)
        Y.append(int(label))
    return X, Y


# distinguish case and control so that we may let each training batch to random sample data from them
def prepare_train_data(labels, icd_codes, med_codes, times, exts, icd2id, med2id,
                       max_visit_len, max_icd_len, max_med_len):
    CASE, CONTROL = [], []
    # print (times)
    # traverse each patient
    for i in range(len(labels)):
        label = int(labels[i])
        icds = icd_codes[i]
        meds = med_codes[i]
        patient_time_infos = times[i]  # three time features for each visit
        patient_ext = exts[i]  # three demographic features for each patient
        flag = 0
        if len(icds) > max_visit_len or len(meds) > max_visit_len or len(patient_time_infos) > max_visit_len:
            flag = 1
        else:
            for sentence in icds:
                if len(sentence) > max_icd_len:
                    flag = 1
                    break
            for sentence in meds:
                if len(sentence) > max_med_len:
                    flag = 1
                    break
        # over the maximum length, then pass
        if flag == 1:
            continue
        doc = make_doc(icds, meds, patient_time_infos, patient_ext,
                       icd2id, med2id, max_visit_len, max_icd_len, max_med_len)
        if label == 1:
            CASE.append([doc, label])
        else:
            CONTROL.append([doc, label])
    return CASE, CONTROL


def random_select(DATA, size):
    random.shuffle(DATA)
    return DATA[0:size]


def train():
    # load train and test data from disk for 1 fold of cross validation
    """
    train_labels: a list of integers
    train_icd/med_codes/times: a list (patients) of list (patient) of list (visit)
    train_exts: a list (patients) of list (patient)
    """
    with open(FLAGS.training_path, 'rb') as data_f:
        train_labels, train_icd_codes, train_med_codes, train_times, train_exts, \
        test_labels, test_icd_codes, test_med_codes, test_times, test_exts = pickle.load(data_f)
    # load dictionaries from disk
    """
    icd/med_vocab: distinct icd/medications
    icd/med Embeddings: if not pre-trained, randomize embeddings
    icd/med2id: from icd/medications to index
    """
    with open(FLAGS.dict_path, 'rb') as dict_f:
        icd_vocab, med_vocab, \
        icd_Embedding, med_Embedding, \
        icd2id, med2id = pickle.load(dict_f)

    train_CASE, train_CONTROL = prepare_train_data(train_labels, train_icd_codes, train_med_codes, train_times,
                                                   train_exts, icd2id, med2id, FLAGS.sequence_length, FLAGS.icd_length,
                                                   FLAGS.med_length)
    testX, testY = prepare_test_data(test_labels, test_icd_codes, test_med_codes, test_times, test_exts, icd2id, med2id,
                                     FLAGS.sequence_length, FLAGS.icd_length, FLAGS.med_length)

    num_of_train = len(train_CASE + train_CONTROL)
    num_of_test = len(testY)
    icd_vocab_size = len(icd2id)
    med_vocab_size = len(med2id)

    print("Loaded pickle data OK")
    print("Training data contains %d points" % num_of_train)
    print("Testing data contains %d points" % num_of_test)
    print("ICD vocab size %d, med vocab size %d" % (icd_vocab_size, med_vocab_size))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = LSTMPrediction(FLAGS.num_classes,
                               FLAGS.learning_rate,
                               FLAGS.decay_steps,
                               FLAGS.decay_rate,
                               FLAGS.batch_size,
                               FLAGS.sequence_length,
                               FLAGS.external_feature_len,
                               FLAGS.icd_length,
                               FLAGS.med_length,
                               icd_vocab_size,
                               med_vocab_size,
                               FLAGS.icd_embed_size,
                               FLAGS.med_embed_size,
                               FLAGS.visit_encode_size,
                               FLAGS.hidden_size,
                               FLAGS.is_training)
    assign_icd_embedding = tf.assign(model.ICD_Embedding, icd_Embedding)
    sess.run(assign_icd_embedding)
    assign_med_embedding = tf.assign(model.MED_Embedding, med_Embedding)
    sess.run(assign_med_embedding)
    print("Building model OK!")

    saver = tf.train.Saver()
    # load model
    if os.path.exists(FLAGS.ckpt_dir + "checkpoint"):
        print("Restoring Variables from Checkpoint for the model.")
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
    else:
        print("Initializing Variables")
        sess.run(tf.global_variables_initializer())

    flag = 0
    max_test_auc = 0.0
    fw_result = open(FLAGS.ckpt_dir + 'results', 'a')

    curr_epoch = sess.run(model.epoch_step)
    batch_size = FLAGS.batch_size

    loss, acc, counter, auc = 0.0, 0.0, 0, 0.0

    for epoch in range(curr_epoch, FLAGS.num_epochs):
        if epoch % 100 == 0:
            print("=" * 30)
            print("Training on batch %d" % epoch)
        sys.stdout.write(str(counter) + '\r')
        sys.stdout.flush()
        # let the case-control be balanced in a training batch
        batch_CASE = random_select(train_CASE, int(batch_size / 2))
        batch_CONTROL = random_select(train_CONTROL, int(batch_size / 2))
        batch_DATA = batch_CASE + batch_CONTROL
        random.shuffle(batch_DATA)
        batch_X = [x[0] for x in batch_DATA]
        batch_Y = [x[1] for x in batch_DATA]
        med_begin = FLAGS.icd_length
        time_begin = FLAGS.icd_length + FLAGS.med_length
        batch_icd = [[visit[0:med_begin] for visit in batch_X[j][:-1]] for j in range(batch_size)]
        batch_med = [[visit[med_begin:time_begin] for visit in batch_X[j][:-1]] for j in range(batch_size)]
        batch_time_features = [[visit[time_begin:] for visit in batch_X[j][:-1]] for j in range(batch_size)]
        batch_ext = [batch_X[j][-1] for j in range(batch_size)]

        curr_loss, curr_acc, _ = sess.run([model.loss_val,
                                           model.accuracy,
                                           model.train_op],
                                          feed_dict={
                                              model.input_x_icd: batch_icd,
                                              model.input_x_med: batch_med,
                                              model.input_x_time: batch_time_features,
                                              model.input_x_ext: batch_ext,
                                              model.input_y: batch_Y,
                                              model.dropout_keep_prob: 0.5
                                          }
                                          )
        loss, acc, counter = loss + curr_loss, acc + curr_acc, counter + 1

        sess.run(model.epoch_increment)
        if epoch % 100 == 0:
            print("Epoch %d [Train loss:%.3f], [Train Accuracy:%.3f]"
                  % (epoch, loss / float(counter), acc / float(counter)))
            test_loss, test_acc, test_auc, output_test = evaluate(sess, model, testX, testY, batch_size)
            if test_auc > max_test_auc:
                max_test_auc = test_auc
            print("Epoch %d Test loss:%.4f, Test Accuracy:%.4f, Test Auc:%.4f" % (epoch, test_loss, test_acc, test_auc))

            # store evaluation performance
            output_test += 'Epoch ' + str(epoch) + ' Test loss:' + str(test_loss) + ' Test acc:' + str(
                test_acc) + ' Test auc:' + str(test_auc) + '\n'
            fw_result.write(output_test)
            fw_result.flush()

            # save model to checkpoint
            save_path = FLAGS.ckpt_dir + '/Epoch_' + str(epoch) + '.model'
            if not os.path.exists(FLAGS.ckpt_dir):
                os.mkdir(FLAGS.ckpt_dir)
            saver.save(sess, save_path, global_step=epoch)
            loss, acc, counter, auc = 0.0, 0.0, 0, 0.0

# this is an evaluation function during training
def evaluate(sess, model, evalX, evalY, batch_size):
    num_of_eval = len(evalY)
    eval_loss, eval_acc, eval_counter, eval_auc = 0.0, 0.0, 0, 0.0
    y_true, y_pred = [], []
    y_probs = []
    output = ''
    for start, end in zip(range(0, num_of_eval, batch_size), range(batch_size, num_of_eval + 1, batch_size)):
        med_begin = FLAGS.icd_length
        time_begin = FLAGS.icd_length + FLAGS.med_length
        batch_icd = [[visit[0:med_begin] for visit in evalX[j][:-1]] for j in range(start, end)]
        batch_med = [[visit[med_begin:time_begin] for visit in evalX[j][:-1]] for j in range(start, end)]
        batch_time_features = [[visit[time_begin:] for visit in evalX[j][:-1]] for j in range(start, end)]
        batch_ext = [evalX[j][-1] for j in range(start, end)]
        curr_loss, curr_acc, predictions, probs = sess.run([model.loss_val, model.accuracy,
                                                            tf.cast(model.predictions, tf.int32),
                                                            model.probs],
                                                           feed_dict={
                                                               model.input_x_icd: batch_icd,
                                                               model.input_x_med: batch_med,
                                                               model.input_x_time: batch_time_features,
                                                               model.input_x_ext: batch_ext,
                                                               model.input_y: evalY[start:end],
                                                               model.dropout_keep_prob: 1.0
                                                           }
                                                           )
        eval_loss, eval_acc, eval_counter = eval_loss + curr_loss, eval_acc + curr_acc, eval_counter + 1
        y_true.extend(evalY[start:end])
        y_pred.extend(predictions)
        y_probs.extend(probs)

    output += "Precision, Recall F1 are %.4f, %.4f, %.4f " + str(precision_score(y_true, y_pred)) + '\t' \
              + str(recall_score(y_true, y_pred)) + '\t' \
              + str(f1_score(y_true, y_pred)) + '\n'
    output += "Confusion matrix is\n"
    output += "-" * 10 + '\n'
    output += str(confusion_matrix(y_true, y_pred)) + '\n'
    output += "-" * 10 + '\n'
    probs = [elem[1] for elem in y_probs]
    fpr, tpr, thresholds = roc_curve(y_true, probs, pos_label=1)
    return eval_loss / float(eval_counter), eval_acc / float(eval_counter), auc(fpr, tpr), output


if __name__ == "__main__":
    # train_random_batch()
    train()
