from abc import abstractclassmethod, abstractstaticmethod


class BaseConverter():

    @abstractclassmethod
    def source_format(self) -> dict:
        pass

    @abstractclassmethod
    def target_format(self) -> dict:
        pass

    @abstractstaticmethod
    def convert(data) -> dict:
        pass
