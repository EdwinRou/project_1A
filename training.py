import data_loader
import network as net

training_data, validation_data, test_data = data_loader.load_data_wrapper()

net_0 = net.Network([784,30,10], net.QuadraticCost)

net_0.SGD(training_data, 10, 10, 0.05, test_data)

net_0.test_feedforward()

net_0.progression()
