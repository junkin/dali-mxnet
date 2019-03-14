from __future__ import print_function
from nvidia.dali.pipeline import Pipeline
import os
import argparse
import fnmatch
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from timeit import default_timer as timer
#get  data set here:
#https://www.kaggle.com/c/dogs-vs-cats
image_dir = "images"
batch_size = 8

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory')
    parser.add_argument('-v', dest='verbose', action='store_true')
    args = parser.parse_args()
    image_dir = args.directory
    speed_test_all(image_dir)

def speed_test_all(dir):
    global image_dir
    image_dir = dir
    pipelines = [SimplePipeline, ShuffledSimplePipeline, RRGPUPipeline, RandomRotatedGPUPipeline, RandomRotatedSimplePipeline]
    test_batch_size = 64
    for i in pipelines:
        speedtest(i, test_batch_size, 16)

def usage():
    print ("-d/--directory specify directory structured like:\ntopdir/kitten/image01.jpg\ntopdir/dog/image02.jpg\n such that you have a top level directory with two subdirs named kitten and dog and in each subdir you have images which are dogs and cats accordingly")


def speedtest(pipeclass, batch, n_threads):
    pipe = pipeclass(batch, n_threads, 0)
    pipe.build()
    # warmup
    for i in range(5):
        pipe.run()
        # test
        n_test = 20
        t_start = timer()
        for i in range(n_test):
            pipe.run()
        t = timer() - t_start
        print("class {}\t Speed: {} imgs/s".format(pipeclass, (n_test * batch)/t))

def list_images():
    for root, dir, files in os.walk("images"):
        depth = root.count("/")
        ret = ""
        if depth > 0:
            ret += " " * (depth - 1) + "|-"
        print (ret + root)
        for items in fnmatch.filter(files, "*"):
                print (" " * len(ret) + "|-" + items)



class SimplePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        print(image_dir)
        super(SimplePipeline, self).__init__(batch_size, num_threads, device_id, seed = 12)
        self.input = ops.FileReader(file_root = image_dir)
        # instead of path to file directory file with pairs image_name image_label_value can be provided
        # self.input = ops.FileReader(file_root = image_dir, file_list = image_dir + '/file_list.txt')
        self.decode = ops.HostDecoder(output_type = types.RGB)

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        return (images, labels)

class ShuffledSimplePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(ShuffledSimplePipeline, self).__init__(batch_size, num_threads, device_id, seed = 12)
        self.input = ops.FileReader(file_root = image_dir, random_shuffle = True, initial_fill = 21)
        self.decode = ops.HostDecoder(output_type = types.RGB)

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        return (images, labels)

class RotatedSimplePipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id):
            super(RotatedSimplePipeline, self).__init__(batch_size, num_threads, device_id, seed = 12)
            self.input = ops.FileReader(file_root = image_dir, random_shuffle = True, initial_fill = 21)
            self.decode = ops.HostDecoder(output_type = types.RGB)
            self.rotate = ops.Rotate(angle = 10.0)

        def define_graph(self):
            jpegs, labels = self.input()
            images = self.decode(jpegs)
            rotated_images = self.rotate(images)
            return (rotated_images, labels)

class RandomRotatedSimplePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(RandomRotatedSimplePipeline, self).__init__(batch_size, num_threads, device_id, seed = 12)
        self.input = ops.FileReader(file_root = image_dir, random_shuffle = True, initial_fill = 21)
        self.decode = ops.HostDecoder(output_type = types.RGB)
        self.rotate = ops.Rotate()
        self.rng = ops.Uniform(range = (-10.0, 10.0))

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        angle = self.rng()
        rotated_images = self.rotate(images, angle = angle)
        return (rotated_images, labels)

class RandomRotatedGPUPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(RandomRotatedGPUPipeline, self).__init__(batch_size, num_threads, device_id, seed = 12)
        self.input = ops.FileReader(file_root = image_dir, random_shuffle = True, initial_fill = 21)
        self.decode = ops.HostDecoder(output_type = types.RGB)
        self.rotate = ops.Rotate(device = "gpu")
        self.rng = ops.Uniform(range = (-10.0, 10.0))

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        angle = self.rng()
        rotated_images = self.rotate(images.gpu(), angle = angle)
        return (rotated_images, labels)

class RRGPUPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(RRGPUPipeline, self).__init__(batch_size, num_threads, device_id, seed = 12)
        self.input = ops.FileReader(file_root = image_dir, random_shuffle = True, initial_fill = 21)
        self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB)
        self.rotate = ops.Rotate(device = "gpu")
        self.rng = ops.Uniform(range = (-10.0, 10.0))

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        angle = self.rng()
        rotated_images = self.rotate(images.gpu(), angle = angle)
        return (rotated_images, labels)


def play():
    pipe = SimplePipeline(batch_size, 1, 0)
    pipe.build()

    pipe_out = pipe.run()
    print(pipe_out)

    images, labels = pipe_out
    print("Images is_dense_tensor: " + str(images.is_dense_tensor()))
    print("Labels is_dense_tensor: " + str(labels.is_dense_tensor()))

    ####### run shuffle pipe versions

    shuffle_pipe = ShuffledSimplePipeline(batch_size, 1, 0)
    shuffle_pipe.build()

    pipe_out = shuffle_pipe.run()
    images, labels = pipe_out
    print("Images is_dense_tensor: " + str(images.is_dense_tensor()))
    print("Labels is_dense_tensor: " + str(labels.is_dense_tensor()))



    ###### run rotated simple pipieline

    rotate_pipe = RotatedSimplePipeline(batch_size, 1, 0)
    rotate_pipe.build()
    pipe_out = rotate_pipe.run()
    images, labels = pipe_out
    print("Images is_dense_tensor: " + str(images.is_dense_tensor()))
    print("Labels is_dense_tensor: " + str(labels.is_dense_tensor()))


    rrgpup = RRGPUPipeline(batch_size, 1, 0)
    rrgpup.build()
    pipe_out = rrgpup.run()
    images, lables = pipe_out
    print("Images is_dense_tensor: " + str(images.is_dense_tensor()))
    print("Labels is_dense_tensor: " + str(labels.is_dense_tensor()))


if __name__ == '__main__':
    main()
