__author__ = 'xiangyang'


"""Predict the possibility of a certain disease at the last step
Features: 
    [visit-level] icd codes + medication name + time features,  [patient-level] race + gender + age
Workflow:       
    for each visit: concatenation(icd9, med, icd_freq, med_freq, time_from_asthma, time_from_previous, time_to_final) -> attention(beta)
    for patient level: visits -> attention(alpha) -> concatenation(visits, race, gender, age)
"""


import tensorflow as tf
from attention import attention
from tensorflow.contrib import rnn

class LSTMPrediction:
    """The model class
    Args:
        sequence_len: the maximum number of visits
        external_feature_len: the size of external features, i.e. demographic features, numeric
        icd_len: the maximum number of icd codes per visit
        med_len: the maximum number of medications per visit
        icd_vocab_size, med_vocab_size: the vocabulary size for icd codes and medications
        icd_embed_size, med_embed_size: the embedding dimension for icd codes and medications
        visit_encode_size, hidden_size: the size for hidden layers
    Returns:

    """
    def __init__(self, num_classes, learning_rate, decay_steps, decay_rate, batch_size,
                 sequence_len, external_feature_len, icd_len, med_len,
                 icd_vocab_size, med_vocab_size,
                 icd_embed_size, med_embed_size,
                 visit_encode_size, hidden_size,
                 is_training, initializer=tf.random_normal_initializer(stddev=0.1)):

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.external_feature_len = external_feature_len
        self.icd_len = icd_len
        self.med_len = med_len
        self.time_feature_len = 3
        self.icd_embed_size = icd_embed_size
        self.med_embed_size = med_embed_size
        self.visit_encode_size = visit_encode_size
        self.hidden_size = hidden_size
        self.is_training = is_training
        self.icd_vocab_size = icd_vocab_size
        self.med_vocab_size = med_vocab_size
        self.lstm_size = self.visit_encode_size + self.time_feature_len
        self.initializer = initializer

        self.input_x_icd = tf.placeholder(tf.int32, [None, self.sequence_len, self.icd_len], name='input_x_icd')
        self.input_x_med = tf.placeholder(tf.int32, [None, self.sequence_len, self.med_len], name='input_x_ndc')

        # [time since asthma index, time since previous visit, time until prediction]
        self.input_x_time = tf.placeholder(tf.float32, [None, self.sequence_len, self.time_feature_len], name='input_x_time')
        # external features: [race, gender, age]
        self.input_x_ext = tf.placeholder(tf.float32, [None, self.external_feature_len], name='input_x_ext')
        self.input_y = tf.placeholder(tf.int32, [None], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        self.global_step = tf.Variable(0, trainable=False, name='Global_step')
        self.epoch_step = tf.Variable(0, trainable=False, name='Epoch_step')
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

        self.initialize_weights()
        self.logits = self.inference()
        self.probs = tf.nn.softmax(self.logits)

        self.loss_val = self.loss()
        self.predictions = tf.argmax(self.logits, axis=1, name='predictions') # shape:[None,] --> batch size
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='Accuracy')

        if is_training:
            self.train_op = self.train()

    def initialize_weights(self):
        with tf.name_scope('Embedding'):
            self.ICD_Embedding = tf.get_variable('ICD_Embedding', shape=[self.icd_vocab_size, self.icd_embed_size], initializer=self.initializer)
            self.MED_Embedding = tf.get_variable('MED_Embedding', shape=[self.med_vocab_size, self.med_embed_size], initializer=self.initializer)

        # visit-level: for hidden layer in each timestamp，the -1 layer
        self.W_hidden = tf.get_variable('W_hidden', shape=[self.lstm_size + self.external_feature_len, self.hidden_size], initializer=self.initializer)
        self.b_hidden = tf.get_variable('b_hidden', shape=[self.hidden_size])

        # for classification, the last layer
        self.W_projection = tf.get_variable('W_projection', shape=[self.hidden_size, self.num_classes], initializer=self.initializer)
        self.b_projection = tf.get_variable('b_projection', shape=[self.num_classes])

        # for code normalization, we let all the input to attention layer as visit_encode_size
        self.W_icd = tf.get_variable('W_icd', shape=[self.icd_embed_size, self.visit_encode_size], initializer=self.initializer)
        self.b_icd = tf.get_variable('b_icd', shape=[self.visit_encode_size])
        self.W_med = tf.get_variable('W_ndc', shape=[self.med_embed_size, self.visit_encode_size], initializer=self.initializer)
        self.b_med = tf.get_variable('b_ndc', shape=[self.visit_encode_size])

    def inference(self):
        # embedding look up
        self.embed_icd = tf.nn.embedding_lookup(self.ICD_Embedding, self.input_x_icd) #[batch_size, sequence_length, icd_length, embed_dim]
        self.embed_med = tf.nn.embedding_lookup(self.MED_Embedding, self.input_x_med) #[batch_size, sequence_length, ndc_length, embed_dim]

        # normalization icd and med to meet the same dimension of event-level attention
        embed_icd = tf.reshape(self.embed_icd, [-1, self.icd_embed_size])
        normal_icd = tf.matmul(embed_icd, self.W_icd) + self.b_icd
        normal_icd = tf.reshape(normal_icd, [-1, self.sequence_len, self.icd_len, self.visit_encode_size])
        embed_med = tf.reshape(self.embed_med, [-1, self.med_embed_size])
        normal_med = tf.matmul(embed_med, self.W_med) + self.b_med
        normal_med = tf.reshape(normal_med, [-1, self.sequence_len, self.med_len, self.visit_encode_size])

        ''' visit-level attention, attention across multiple medical events '''
        # combine icd, ndc as the visit
        self.events = tf.concat([normal_icd, normal_med], axis=2) # [batch_size, sequence_length, icd_length+ndc_length, visit_encode_size]
        # reshape to meet the 3D input of attention
        self.events = tf.reshape(self.events, [self.batch_size*self.sequence_len, self.icd_len+self.med_len, self.visit_encode_size])
        atten_events, betas = attention(self.events, self.visit_encode_size, time_major=False, return_alphas=True)
        atten_events = tf.reshape(atten_events, [self.batch_size, self.sequence_len, self.visit_encode_size])
        # for each event (icd or ndc), there is an attention weight to denote the attribution
        self.betas = tf.reshape(betas, [self.batch_size, self.sequence_len, self.icd_len+self.med_len])
        self.visits = tf.concat([atten_events, self.input_x_time], axis=2) # [batch_size, sequence_length, visit_encode_size+time_length]

        ''' patient-level attention LSTM '''
        with tf.name_scope('visit_lstm_layer'):
            lstm_cell = rnn.BasicLSTMCell(self.lstm_size)
        if self.dropout_keep_prob != 0.0:
            lstm_cell = rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_keep_prob)
        # patient_lstm_outputs: [batch_size, sequence_len, lstm_size]
        patient_lstm_outputs, patients_states = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=self.visits, dtype='float32')
        # patient_lstm_outputs: [batch_size, lstm_size], alphas: [batch_size, sequence_len]
        patient_lstm_outputs, alphas = attention(patient_lstm_outputs, self.lstm_size, time_major=False, return_alphas=True)
        self.alphas = alphas
        fc_layer_inputs = tf.concat([patient_lstm_outputs, self.input_x_ext], axis=1)

        # fully connected layer, MLP
        with tf.name_scope('fc_layer'):
            fc_layer = tf.matmul(fc_layer_inputs, self.W_hidden) + self.b_hidden
            # batch normalization
            fc_layer_bn = tf.contrib.layers.batch_norm(fc_layer, center=True, scale=True, is_training=self.is_training, scope='bn')
            logits = tf.matmul(fc_layer_bn, self.W_projection) + self.b_projection
        return logits

    # losses for each trainable variable
    def loss(self, l2_lambda=0.0001):
        with tf.name_scope('loss'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss += l2_losses
        return loss

    # the training operation
    def train(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step, learning_rate=learning_rate, optimizer='Adam')
        return train_op
