"""
TODO: docstring
"""
import copy
import MANN
import numpy
import os
import tensorflow
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Omniglot:
    """
    TODO: docstring
    """
    def omniglot(self):
        """
        TODO: docstring
        """
        sess = tensorflow.InteractiveSession()
        input_ph = tensorflow.placeholder(dtype=tensorflow.float32, shape=(16, 50, 400))
        target_ph = tensorflow.placeholder(dtype=tensorflow.int32, shape=(16, 50))
        nb_reads = 4
        controller_size = 200
        memory_shape = (128, 40)
        nb_class = 5
        input_size = 20*20
        batch_size = 16
        nb_samples_per_class = 10
        generator = MANN.Utils.Generator.OmniglotGenerator(
            data_folder='./data/omniglot', batch_size=batch_size, nb_samples=nb_class,
            nb_samples_per_class=nb_samples_per_class, max_rotation=0.0, max_shift=0.0, max_iter=None)
        output_var, output_var_flatten, params = MANN.Model.memory_augmented_neural_network(
            input_ph, target_ph, batch_size=batch_size, nb_class=nb_class, memory_shape=memory_shape,
            controller_size=controller_size, input_size=input_size, nb_reads=nb_reads)
        print('Compiling the Model')
        with tensorflow.variable_scope("Weights", reuse=True):
            W_key = tensorflow.get_variable('W_key', shape=(nb_reads, controller_size, memory_shape[1]))
            b_key = tensorflow.get_variable('b_key', shape=(nb_reads, memory_shape[1]))
            W_add = tensorflow.get_variable('W_add', shape=(nb_reads, controller_size, memory_shape[1]))
            b_add = tensorflow.get_variable('b_add', shape=(nb_reads, memory_shape[1]))
            W_sigma = tensorflow.get_variable('W_sigma', shape=(nb_reads, controller_size, 1))
            b_sigma = tensorflow.get_variable('b_sigma', shape=(nb_reads, 1))
            W_xh = tensorflow.get_variable('W_xh', shape=(input_size + nb_class, 4 * controller_size))
            b_h = tensorflow.get_variable('b_xh', shape=(4 * controller_size))
            W_o = tensorflow.get_variable('W_o', shape=(controller_size + nb_reads * memory_shape[1], nb_class))
            b_o = tensorflow.get_variable('b_o', shape=(nb_class))
            W_rh = tensorflow.get_variable('W_rh', shape=(nb_reads * memory_shape[1], 4 * controller_size))
            W_hh = tensorflow.get_variable('W_hh', shape=(controller_size, 4 * controller_size))
            gamma = tensorflow.get_variable('gamma', shape=[1], initializer=tensorflow.constant_initializer(0.95))
        params = [W_key, b_key, W_add, b_add, W_sigma, b_sigma, W_xh, W_rh, W_hh, b_h, W_o, b_o]
        target_ph_oh = tensorflow.one_hot(target_ph, depth=generator.nb_samples)
        print('Output, Target shapes: ', output_var.get_shape().as_list(), target_ph_oh.get_shape().as_list())
        cost = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits(
            output_var, target_ph_oh), name="cost")
        opt = tensorflow.train.AdamOptimizer(learning_rate=1e-3)
        train_step = opt.minimize(cost, var_list=params)
        accuracies = MANN.Utils.Metrics.accuracy_instance(tensorflow.argmax(
            output_var, axis=2), target_ph, batch_size=generator.batch_size)
        sum_out = tensorflow.reduce_sum(tensorflow.reshape(tensorflow.one_hot(tensorflow.argmax(
            output_var, axis=2), depth=generator.nb_samples), (-1, generator.nb_samples)), axis=0)
        print('Done')
        tensorflow.summary.scalar('cost', cost)
        for i in range(generator.nb_samples_per_class):
            tensorflow.summary.scalar('accuracy-' + str(i), accuracies[i])
        merged = tensorflow.summary.merge_all()
        train_writer = tensorflow.summary.FileWriter('/tmp/tensorflow/', sess.graph)
        t0 = time.time()
        all_scores, scores, accs = [],[],numpy.zeros(generator.nb_samples_per_class)
        sess.run(tensorflow.global_variables_initializer())
        print('Training the model')
        try:
            for i, (batch_input, batch_output) in generator:
                feed_dict = {input_ph: batch_input, target_ph: batch_output}
                train_step.run(feed_dict)
                score = cost.eval(feed_dict)
                acc = accuracies.eval(feed_dict)
                temp = sum_out.eval(feed_dict)
                summary = merged.eval(feed_dict)
                train_writer.add_summary(summary, i)
                print(i , ' ', temp)
                all_scores.append(score)
                scores.append(score)
                accs += acc
                if i > 0 and not (i % 100):
                    print(accs / 100.0)
                    print('Episode %05d: %.6f' % (i, numpy.mean(score)))
                    scores, accs = [], numpy.zeros(generator.nb_samples_per_class)
        except KeyboardInterrupt:
            print(time.time() - t0)
            pass

class TestUPD:
    """
    TODO: docstring
    """
    def omniglot():
        sess = tensorflow.InteractiveSession()
        #v = tensorflow.Variable(
        #    initial_value=numpy.arange(0, 36).reshape((6, 6)), dtype=tensorflow.float32, name='Matrix')
        #sess.run(tensorflow.global_variables_initializer())
        #sess.run(tensorflow.local_variables_initializer())
        #temp = tensorflow.Variable(
        #    initial_value=numpy.arange(0, 36).reshape((6, 6)), dtype=tensorflow.float32, name='temp')
        #temp = wrapper(v)
        #temp.eval()
        #print('Hello')
        #self.update_tensor()
        print('Compiling the Model')
        tt1 = tensorflow.Variable(
            initial_value=numpy.arange(0, 36).reshape((6, 6)), dtype=tensorflow.float32, name='Matrix')
        ix = tensorflow.Variable(initial_value=numpy.arange(0, 6), name='Indices')
        val = tensorflow.Variable(initial_value=numpy.arange(100, 106), name='Values', dtype=tensorflow.float32)
        tt = tensorflow.concat_v2([tt1[:3], tensorflow.reshape(
            tensorflow.range(0,6,dtype=tensorflow.float32),shape=(1,6)), tt1[3:]], axis=0)
        print(tt1[:3].get_shape().as_list())
        #op = tt1[4].assign(val)
        #sess.run(tensorflow.global_variables_initializer())
        #sess.run(op)
        #print(tt1.eval())
        op = tt1.assign(MANN.Utils.tf_utils.update_tensor(tt1, ix, val))
        val = tensorflow.Print(val, [val], "This works fine")
        sess.run(tensorflow.global_variables_initializer())
        #sess.run(tensorflow.local_variables_initializer())
        print('Training the model')
        print(tt.eval())
        writer = tensorflow.summary.FileWriter('/tmp/tensorflow', graph=tensorflow.get_default_graph())
        #tensorflow.scalar_summary('cost', cost)
        print('tt1: ',tt1.eval())
        print('ix: ',ix.eval())
        print('val: ',val.eval())
        sess.run(op)
        print('After run\n', tt1.eval())
        #with tensorflow.control_dependencies([op]):
            #print(tt1.eval(), '\n', op.eval())
            
    def body(_, v, d2, chg):
        """
        TODO: docstring
        """
        d2_int = tensorflow.cast(d2, tensorflow.int32)
        return tensorflow.slice(tensorflow.concat_v2(
            [v[:d2_int], [chg], v[d2_int+1:]], axis=0), [0], [v.get_shape().as_list()[0]])

    def update_tensor(V, dim2, val):
        """
        Update tensor V, with index(:, dim2[:]) by val[:].
        """
        val = tensorflow.cast(val, V.dtype)
        self.body(V, dim2, val)
        Z = tensorflow.scan(
            body, elems=(V, dim2, val), initializer=tensorflow.constant(
                1, shape=V.get_shape().as_list()[1:], dtype=tensorflow.float32), name="Scan_Update")
        return Z

    def wrapper(v):
        """
        TODO: docstring
        """
        return tensorflow.Print(v, [v], message="Printing v")

class Testing:
    """
    TODO: docstring
    """
    def __call__(self):
        """
        TODO: docstring
        """
        x = [0, 0, 0, 0, 0] * 10
        y = [0, 1, 2, 3, 4] * 10
        numpy.random.shuffle(y)
        x = numpy.append([x],[x],axis=0)
        y = numpy.append([y], [y], axis=0)
        p = tensorflow.constant(x)
        t = tensorflow.constant(y)
        sess = tensorflow.InteractiveSession()
        zz = MANN.Utils.Metrics.accuracy_instance(p, t, batch_size=2)
        sess.run(zz)
        print(p[0].eval())
        print(t[0].eval())
        print(zz.eval())
        print()

    def omniglot(self):
        """
        TODO: docstring
        """
        sess = tensorflow.InteractiveSession()
        input_ph = tensorflow.placeholder(dtype=tensorflow.float32, shape=(16,50,400))   #(batch_size, time, input_dim)
        target_ph = tensorflow.placeholder(dtype=tensorflow.int32, shape=(16,50))     #(batch_size, time)(label_indices)
        nb_reads = 4
        controller_size = 200
        memory_shape = (128,40)
        nb_class = 5
        input_size = 20*20
        batch_size = 16
        nb_samples_per_class = 10
        generator = MANN.Utils.Generator.OmniglotGenerator(
            data_folder='./data/omniglot', batch_size=batch_size, nb_samples=nb_class,
            nb_samples_per_class=nb_samples_per_class, max_rotation=0., max_shift=0., max_iter=None)
        output_var, output_var_flatten, params = MANN.Model.memory_augmented_neural_network(
            input_ph, target_ph, batch_size=batch_size, nb_class=nb_class, memory_shape=memory_shape,
            controller_size=controller_size, input_size=input_size, nb_reads=nb_reads)
        print('Compiling the Model')
        with tensorflow.variable_scope('Weights', reuse=True):
            W_key = tensorflow.get_variable('W_key', shape=(nb_reads, controller_size, memory_shape[1]))
            b_key = tensorflow.get_variable('b_key', shape=(nb_reads, memory_shape[1]))
            W_add = tensorflow.get_variable('W_add', shape=(nb_reads, controller_size, memory_shape[1]))
            b_add = tensorflow.get_variable('b_add', shape=(nb_reads, memory_shape[1]))
            W_sigma = tensorflow.get_variable('W_sigma', shape=(nb_reads, controller_size, 1))
            b_sigma = tensorflow.get_variable('b_sigma', shape=(nb_reads, 1))
            W_xh = tensorflow.get_variable('W_xh', shape=(input_size + nb_class, 4 * controller_size))
            b_h = tensorflow.get_variable('b_xh', shape=(4 * controller_size))
            W_o = tensorflow.get_variable('W_o', shape=(controller_size + nb_reads * memory_shape[1], nb_class))
            b_o = tensorflow.get_variable('b_o', shape=(nb_class))
            W_rh = tensorflow.get_variable(
                'W_rh', shape=(nb_reads * memory_shape[1], 4 * controller_size))
            W_hh = tensorflow.get_variable('W_hh', shape=(controller_size, 4 * controller_size))
            gamma = tensorflow.get_variable(
                'gamma', shape=[1], initializer=tensorflow.constant_initializer(0.95))
        params = [W_key, b_key, W_add, b_add, W_sigma, b_sigma, W_xh, W_rh, W_hh, b_h, W_o, b_o]
        #output_var = tensorflow.cast(output_var, tensorflow.int32)
        target_ph_oh = tensorflow.one_hot(target_ph, depth=generator.nb_samples)
        print('Output, Target shapes: ',
              output_var.get_shape().as_list(), target_ph_oh.get_shape().as_list())
        cost = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits(
            output_var, target_ph_oh), name="cost")
        opt = tensorflow.train.AdamOptimizer(learning_rate=1e-3)
        train_step = opt.minimize(cost, var_list=params)
        #train_step = tensorflow.train.AdamOptimizer(1e-3).minimize(cost)
        accuracies = MANN.Utils.Metrics.accuracy_instance(
            tensorflow.argmax(output_var, axis=2), target_ph, batch_size=generator.batch_size)
        sum_out = tensorflow.reduce_sum(tensorflow.reshape(tensorflow.one_hot(tensorflow.argmax(
            output_var, axis=2), depth=generator.nb_samples), (-1, generator.nb_samples)), axis=0)
        print('Done')
        tensorflow.summary.scalar('cost', cost)
        for i in range(generator.nb_samples_per_class):
            tensorflow.summary.scalar('accuracy-'+str(i), accuracies[i])
        merged = tensorflow.summary.merge_all()
        #writer = tensorflow.summary.FileWriter('/tmp/tensorflow', graph=tensorflow.get_default_graph())
        train_writer = tensorflow.summary.FileWriter('/tmp/tensorflow/', sess.graph)
        t0 = time.time()
        all_scores, scores, accs = [], [], numpy.zeros(generator.nb_samples_per_class)
        sess.run(tensorflow.global_variables_initializer())
        print('Training the model')
        try:
            for i, (batch_input, batch_output) in generator:
                feed_dict = {input_ph: batch_input, target_ph: batch_output}
                #print(batch_input.shape, batch_output.shape)
                train_step.run(feed_dict)
                score = cost.eval(feed_dict)
                acc = accuracies.eval(feed_dict)
                temp = sum_out.eval(feed_dict)
                summary = merged.eval(feed_dict)
                train_writer.add_summary(summary, i)
                print(i, ' ', temp)
                all_scores.append(score)
                scores.append(score)
                accs += acc
                if i > 0 and not i % 100:
                    print(accs / 100.0)
                    print('Episode %05d: %.6f' % (i, numpy.mean(score)))
                    scores, accs = [], numpy.zeros(generator.nb_samples_per_class)
        except KeyboardInterrupt:
            print(time.time() - t0)
            pass

if __name__ == '__main__':
    Omniglot.omniglot()
