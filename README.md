# flower-async
Implementation of asynchronous federated learning in flower currently being developed as a part of my master's thesis. 

Currently offers the following modes of operation:
- Unweighted - Upon each merge into the global model, the local updates and global model are weighted equally.
- FedAsync - global_new = (1-alpha) * global_old + alpha * local_new
  - Where alpha depends on the staleness and/or number of samples used
- AsyncFedED - global_new = global_old + $\eta$ * (local_new - global_old)
  - $\eta$ again depends on staleness and/or number of samples

# Implementation

The implementation consists of a modified server, strategy and client manager.

The inspiration of this work stems from the following papers:
- [Privacy-Preserving Asynchronous Federated Learning Mechanism for Edge Network Computing -  ](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9022982)
- [Asynchronous Federated Optimization ~ FedAsync](https://arxiv.org/pdf/1903.03934.pdf)
- [AsyncFedED: Asynchronous Federated Learning with Euclidean Distance based Adaptive Weight Aggregation](https://arxiv.org/pdf/2205.13797.pdf)
- [Federated Learning on Non-IID Data Silos: An Experimental Study](https://arxiv.org/abs/2102.02079)

# Usage

Use the server as typical flower server with one additional argument: 
- async_strategy - currently only responsible for aggregating the models.

Moreover the async client manager should be used instead of the SimpleClientManager. 

The server instantiation would hence look something like this:
```
server = AsyncServer(
    strategy=FedAvg(<customized>), 
    client_manager=AsyncClientManager(), 
    async_strategy=AsynchronousStrategy(<customized>))
```
