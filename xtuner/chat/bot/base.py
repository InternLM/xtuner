from abc import abstractmethod


class BaseBot():

    @property
    def generation_config(self):
        pass

    @abstractmethod
    def create_streamer(self, iterable=False):
        pass

    @abstractmethod
    def generate(self, inputs, streamer=None, generation_config=None):
        pass

    @abstractmethod
    def predict(self, inputs, generation_config=None, repeat=1):
        pass


class BaseLlavaBot():

    @abstractmethod
    def process_img(self, image):
        pass

    @abstractmethod
    def prepare_inputs(self, text, pixel_values, n_turn):
        pass

    @property
    def generation_config(self):
        pass

    @abstractmethod
    def create_streamer(self, iterable=False):
        pass

    @abstractmethod
    def generate(self, inputs, streamer=None, generation_config=None):
        pass

    @abstractmethod
    def predict(self, inputs, generation_config=None, repeat=1):
        pass
