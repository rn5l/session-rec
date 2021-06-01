<h1>session-rec</h1>
<h2>Introduction</h2>

<b>session-rec</b> is a Python-based framework for building and evaluating recommender systems (Python 3.5.x). It
implements a suite of state-of-the-art algorithms and baselines for session-based and session-aware recommendation.
<br><br>
The authors developed this framework to carry out the experiments described in:
<ul>
    <li>
        S. Latifi, N. Mauro and D. Jannach. 2021. Session-aware recommendation: a surprising quest for the state-of-the-art. Information Sciences.
    </li>
    <li>
        M. Ludewig, N. Mauro, S. Latifi and D. Jannach. 2020. Empirical analysis of session-based recommendation algorithms. User Modeling and User-Adapted Interaction 31 (1), 149-181. 
    </li>
    <li>
        M. Ludewig, N. Mauro, S. Latifi and D. Jannach. Performance comparison of neural and non-neural approaches to session-based recommendation. 2019. Proceedings of the 13th ACM conference on recommender systems, 462-466. 
    </li>
    <li>
         M. Ludewig and D. Jannach. Evaluation of session-based recommendation algorithms. 2018. User Modeling and User-Adapted Interaction 28 (4-5), 331-390. 
    </li>
</ul>
Parts of the framework and its algorithms are based on code developed and shared by:
<ul>
    <li>
        Quadrana et al., Personalizing Session-based Recommendations with Hierarchical Recurrent Neural Networks, RecSys 2017. <a
            href="https://github.com/mquad/hgru4rec">(Original Code).</a>
    </li>
    <li>
        Ruocco et al., Inter-session modeling for session-based recommendation, DLRS 2017. <a
            href="https://github.com/rainmilk/ieee-is-ncsf">(Original Code).</a>
    </li>
    <li>
        Ying et al., Sequential recommender system based on hierarchical attention network, IJCAI 2018. <a
            href="https://github.com/chenghu17/Sequential_Recommendation/tree/master/SHAN">(Original Code).</a>
    </li>
    <li>
        Liang et al., Neural cross-session filtering: Next-item prediction under intra- and inter-session context, IEEE Intelligent Systems 2019. <a
            href="https://github.com/rainmilk/ieee-is-ncsf">(Original Code).</a>
    </li> 
    <li>
        Phuong et al., Neural session-aware recommendation, IEEE Access 2019. <a
            href="https://github.com/thanhtcptit/RNN-for-Resys">(Original Code).</a>
    </li>    
    <li>
        Rendle et al., BPR: Bayesian Personalized Ranking from Implicit Feedback, UAI 2009. <a
            href="https://github.com/hidasib/GRU4Rec/blob/master/baselines.py">(Original Code).</a>
    </li>
    <li>
        Mi et al., Context Tree for Adaptive Session-based Recommendation, 2018. (Code shared by the authors).
    </li>
    <li>
        Hidasi et al., Recurrent Neural Networks with Top-k Gains for Session-based Recommendations, CoRR
        abs/1706.03847, 2017. <a href="https://github.com/hidasib/GRU4Rec">(Original Code).</a>
    </li>
    <li>
        Liu et al., STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation, KDD
        2018. <a href="https://github.com/uestcnlp/STAMP">(Original Code).</a>
    </li>
    <li>
        Li et al., Neural Attentive Session-based Recommendation, CIKM 2017. <a
            href="https://github.com/lijingsdu/sessionRec_NARM">(Original Code).</a>
    </li>
    <li>
        Yuan et al., A Simple but Hard-to-Beat Baseline for Session-based Recommendations, CoRR
        abs/1808.05163, 2018. (Code shared by the authors).
    </li>
    <li>
        Wu et al., Session-based recommendation with graph neural networks, AAAI, 2019. <a
            href="https://github.com/CRIPAC-DIG/SR-GNN">(Original Code).</a>
    </li>
    <li>
        Wang et al., A collaborative session-based recommendation approach with parallel memory modules, SIGIR, 2019. <a
            href="https://github.com/wmeirui/CSRM_SIGIR2019">(Original Code).</a>
    </li>
    <li>
        Rendle et al., Factorizing Personalized Markov Chains for Next-basket Recommendation. WWW 2010. <a
            href="https://github.com/rdevooght/sequence-based-recommendations/blob/master/factorization/fpmc.py">(Original
        Code).</a>
    </li>
    <li>
        Kabbur et al., FISM: Factored Item Similarity Models for top-N Recommender Systems, KDD 2013. <a
            href="https://github.com/rdevooght/sequence-based-recommendations/blob/master/factorization/fism.py">(Original
        Code).</a>
    </li>
    <li>
        He and McAuley. Fusing Similarity Models with Markov Chains for Sparse Sequential Recommendation.
        CoRR abs/1609.09152, 2016. <a
            href="https://github.com/rdevooght/sequence-based-recommendations/blob/master/factorization/fossil.py">(Original
        Code).</a>
    </li>
</ul>


<h2>Requirements</h2>
To run session-aware, the following libraries are required:
<ul>
    <li>Anaconda 4.X (Python 3.5)</li>
    <li>Pympler</li>
    <li>NumPy</li>
    <li>SciPy</li>
    <li>BLAS</li>
    <li>Sklearn</li>
    <li>Dill</li>
    <li>Pandas</li>
    <li>Theano</li>
    <li>Pyyaml</li>
    <li>CUDA</li>
    <li>Tensorflow</li>
    <li>Theano</li>
    <li>Psutil</li>
    <li>Scikit-learn</li>
    <li>Tensorflow-gpu</li>
    <li>NetworkX</li>
    <li>Certifi</li> 
    <li>NumExpr</li>
    <li>Pytable</li>
    <li>Python-dateutil</li>
    <li>Pytz</li>
    <li>Six</li>
    <li>Keras</li>
    <li>Scikit-optimize</li>
    <li>Python-telegram-bot</li>
</ul>

<h2>Installation</h2>
<h3>
    Using docker 
</h3>
<ol>
    <li>Download and Install Docker (https://www.docker.com/)</li>
    <li>Run the following commands:
        <ol>
            <li>If you are using Windows or you are using Linux but you don't have a GPU: <code>docker pull maltel/session-rec-cpu:v1</code></li>
            <li>If you are using LINUX and you have a GPU: <code>docker pull maltel/session-rec-gpu:v1</code></li>
        </ol>
    </li>
    <li>Download the repository: <code>https://github.com/rn5l/session-rec.git</code></li>
</ol>
<h3>
    Using Anaconda
</h3>
<ol>
    <li>Download and Install Anaconda (https://www.anaconda.com/distribution/)</li>
    <li>Run the following command:<br>
            From the main folder run: </br>
        <ol>
            <li>If you have a GPU: <code>conda install --yes --file environment_gpu.yml</code></li>
            <li>If you don't have a GPU or you are using Windows: <code>conda install --yes --file environment_cpu.yml</code></li>  
        </ol>
    </li>
    <li>Activate the conda environment: <code>conda activate srec37</code></li>
    <li>Download the repository: <code>https://github.com/rn5l/session-rec.git</code></li>
</ol>
<h2>Example of Experiments</h2>
The data folder contains a small sample dataset. It's possible to have an overview of how the framework works by using as a configuration file:
    <ul>
        <li>
            example_next.yml to predict the next item in the session. 
        </li>
        <li>
            example_multiple.yml to predict the remaining items of the session.
        </li>
    </ul>
At the end of the experiments, you can find the evalutaion results in the "results" folder. You can also find the list of recommended items under the "results" folder with the suffix "Saver@". 
<h2>How to Run It</h2>
<ol>
    <h3>
        <li>Run experiments using the configuration file
    </h3>
    <ol>
        <li>
            Create folders conf/in and conf/out. Configure a configuration file <b>*.yml</b> and put it into the folder named conf/in. Examples of configuration
            files are listed in the conf folder. It is possible to configure multiple files and put them all in
            the conf/in folder. When a configuration file in conf/in has been executed, it will be moved to the folder conf/out.
        </li>
        <li>
            <b>Using Docker: </b> </br>
            Run the following command from the main folder: </br>
            <ol>
            <li> If you are using Linux and you have a GPU:
            <code>
                ./dpython_gpu run_config.py conf/in conf/out
            </code> 
            </li>
            <li> If you are using Linux and you don't have a GPU:
            <code>
                ./dpython run_config.py conf/in conf/out
            </code> 
            </li>
            <li> If you are using Windows:
            <code>
                ./dpython.bat run_config.py conf/in conf/out
            </code> 
            </li>
        </ol>
        </li>
        <li>
            <b>Using Anaconda:</b> </br>
            Run the following command from the main folder: </br>
            <ol>
                <li>If you are using Linux and you have a GPU:
                <code>
                    THEANO_FLAGS="device=cuda0,floatX=float32" CUDA_DEVICE_ORDER=PCI_BUS_ID python run_config.py conf/in conf/out
                </code>
                </li>
                <li>If you are using Windows or you are using Linux but you don't have a GPU:
                <code>
                    python run_config.py conf/in conf/out
                </code>
                </li>
            </ol>
        </li>
        <li>
            Results and run times will be displayed and saved to the results folder as config.
        </li>
        <li>
            If you want to run a specific configuration file, you have to use:</br>
            <code>
            conf/example_next.yml
            </code>
            instead of:
            <code>
            conf/in conf/out
            </code>
        </li>
    </li>
        </ol>
                <h3>
        <li>Dataset preprocessing
        </h3>
        <ol>
            <li>
                Unzip any dataset file to the data folder, i.e., rsc15-clicks.dat will then be in the folder
                data/rsc15/raw
            </li>
            <li>
                Open and edit any configuration file in the folder conf/preprocess/.. to configure the preprocessing method and parameters.
                <ul>
                    <li>
                        See, e.g., conf/preprocess/window/rsc15.yml for an example with comments.
                    </li>
                </ul>
            </li>
            <li>
                Run a configuration file with the following command using the commands described above based on your OS. For example for Linux users that have a GPU and are using docker: </br>
                <code>
                    ./dpython_gpu run_preprocessing.py conf/preprocess/window/rsc15.yml
                </code>
                </br>
                Otherwise, replace <code> ./dpython_gpu </code> with the correct command based on your installation and your OS.
            </li>
    </ol>
<h2>How to Configure It</h2>
<b>Start from one of the examples in the conf folder.</b>
<h3>Essential Options</h3>
<div>
<div>
    <table class="table table-hover table-bordered">
        <tr>
            <th width="12%" scope="col"> Entry</th>
            <th width="16%" class="conf" scope="col">Example</th>
            <th width="72%" class="conf" scope="col">Description</th>
        </tr>
        <tr>
            <td>type</td>
            <td>window</td>
            <td>Values: single (one single training-test split), window (sliding-window protocol), opt (parameters
                optimization).
            </td>
        </tr>
        <tr>
            <td>evaluation</td>
            <td>evaluation_user_based</td>
            <td>Values: <b>for session-aware evaluation:</b> evaluation_user_based (evaluation in term of the next item and in terms of the remaining items of the sessions),  <b>for session-based evaluation:</b> evaluation (evaluation in term of the next item), evaluation_last (evaluation in term of the last item of the session), evaluation_multiple (evaluation in terms of the remaining items of the sessions).
            </td>
        </tr>
        <tr>
            <td scope="row">slices</td>
            <td>5</td>
            <td>Number of slices for the window protocol.<br>
            </td>
        </tr>
        <tr>
            <td scope="row">opts</td>
            <td>opts: {sessions_test: 10}</td>
            <td>Number of sessions used as a test during the optimization phase.<br>
            </td>
        </tr>
        <tr>
            <td scope="row">metrics</td>
            <td>-class: accuracy.HitRate<br>
                length: [5,10,15,20]
            </td>
            <td>List of accuracy measures (HitRate, MRR, Precision, Recall, MAP, Coverage, Popularity,
                Time_usage_training, Time_usage_testing, Memory_usage).
                If you want to save the files with the recommedation lists use the option: <br>
                <code> - class: saver.Saver<br>
                    length: [50]</code>
                It's possible to use the saved recommendations using the ResultFile class.
            </td>
        </tr>
        <tr>
            <td scope="row">opts</td>
            <td>opts: {sessions_test: 10}</td>
            <td>Number of session used as a test during the optimization phase.<br>
            </td>
        </tr>
        <tr>
            <td scope="row">optimize</td>
            <td> class: accuracy.MRR <br>
                length: [20]<br>
                iterations: 100 #optional
            </td>
            <td>Measure to which optimize the parameters.<br>
            </td>
        </tr>
        <tr>
            <td scope="row">algorithms</td>
            <td>-</td>
            <td>See the configuration files in the conf folder for a list of the
                algorithms and their parameters.<br>
            </td>
        </tr>
    </table>
</div>
<h2>Algorithms</h2>
<div>
    <h3>Baselines</h3>
    <div>
        <table class="table table-hover table-bordered">
            <tr>
                <th width="20%" scope="col"> Algorithm</th>
                <th width="12%" class="conf" scope="col">File</th>
                <th width="68%" class="conf" scope="col">Description</th>
            </tr>
            <tr>
                <td scope="row">Association Rules</td>
                <td>ar.py</td>
                <td>Simplified version of the association rule mining technique with a maximum rule size of two.<br>
                </td>
            </tr>
            <tr>
                <td scope="row">Markov Chains</td>
                <td>markov.py</td>
                <td>Variant of association rules with a focus on sequences in the data. The rules are extracted from a
                    first-order Markov Chain.
                </td>
            </tr>
            <tr>
                <td scope="row">Sequential Rules</td>
                <td>sr.py</td>
                <td>Variation of mc or ar respectively. It also takes the order of actions into account, but in a less
                    restrictive manner.
                </td>
            </tr>
            <tr>
                <td scope="row">BPR-MF</td>
                <td>bpr.py</td>
                <td>Rendle et al., BPR: Bayesian Personalized Ranking from Implicit Feedback, UAI 2009.
                </td>
            </tr>
            <tr>
                <td scope="row">Context Tree</td>
                <td>ct.py</td>
                <td>Mi et al., Context Tree for Adaptive Session-based Recommendation, 2018.
                </td>
            </tr>
        </table>
    </div>
    <h3>Nearest Neighbors</h3>
    <div>
    <table class="table table-hover table-bordered">
        <tr>
            <th width="20%" scope="col"> Algorithm</th>
            <th width="12%" class="conf" scope="col">File</th>
            <th width="68%" class="conf" scope="col">Description</th>
        </tr>
        <tr>
            <td scope="row">Item-based kNN</td>
            <td>iknn.py</td>
            <td>Considers the last element in a given session and then returns those items as recommendations that
                are most similar to it in terms of their co-occurrence in other sessions.<br>
            </td>
        </tr>
        <tr>
            <td scope="row">Session-based kNN</td>
            <td>sknn.py</td>
            <td>
                Recommend items from the most similar sessions, where session distance is determined with the cosine similarity function or the jaccard index. 
            </td>
        </tr>
        <tr>
            <td scope="row">Vector Multiplication Session-Based kNN</td>
            <td>vsknn.py</td>
            <td>More emphasis on the more recent events of a session when computing the similarities. The weights of
                the other elements are determined using a linear decay function that
                depends on the position of the element within the session, where elements appearing earlier in the
                session obtain a lower weight. 
            </td>
        </tr>
        <tr>
            <td scope="row">Sequence and Time Aware Neighborhood</td>
            <td>stan.py</td>
            <td>Garg et al., Sequence and time aware neighborhood for session-based recommendations: Stan, SIGIR 2019.
            </td>
        </tr>
        <tr>
            <td scope="row">Sequence and Time Aware Neighborhood</td>
            <td>vstan.py</td>
            <td>It combines ideas from stan and v-sknn in a single approach. Furthermore, it has a sequence-aware item scoring procedure 
                as well as the IDF weighting scheme from v-sknn.
            </td>
        </tr>
    </table>
    </div>
    <h3>Session-based Neural Models</h3>
    <div>
    <table class="table table-hover table-bordered">
        <tr>
            <th width="20%" scope="col"> Algorithm</th>
            <th width="12%" class="conf" scope="col">File</th>
            <th width="68%" class="conf" scope="col">Description</th>
        </tr>
        <tr>
            <td scope="row">CSRM</td>
            <td>csrm.py</td>
            <td>Wang et al., A collaborative session-based recommendation approach with parallel memory modules, SIGIR 2019.<br>
            </td>
        </tr>
        <tr>
            <td scope="row">Gru4Rec</td>
            <td>gru4rec.py</td>
            <td>Hidasi et al., Recurrent Neural Networks with Top-k Gains for Session-based Recommendations, CIKM 2018.<br>
            </td>
        </tr>
        <tr>
            <td scope="row">NextItNet</td>
            <td>nextitrec.py</td>
            <td>Yuan et al., A simple convolutional generative network for next item recommendation, WSDM 2019.
            </td>
        </tr>
        <tr>
            <td scope="row">NARM</td>
            <td>narm.py</td>
            <td>Li et al., Neural Attentive Session-based Recommendation, CIKM 2017.
            </td>
        </tr>
        <tr>
            <td scope="row">SR-GNN</td>
            <td>gnn.py</td>
            <td>Wu et al., Session-based recommendation with graph neural networks, AAAI 2019.
            </td>
        </tr>
        <tr>
            <td scope="row">STAMP</td>
            <td>STAMP.py</td>
            <td>Liu et al., STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation, KDD
                2018.
            </td>
        </tr>
    </table>
    </div>
    <h3>Session-aware Neural Models</h3>
    <div>
    <table class="table table-hover table-bordered">
        <tr>
            <th width="20%" scope="col"> Algorithm</th>
            <th width="12%" class="conf" scope="col">File</th>
            <th width="68%" class="conf" scope="col">Description</th>
        </tr>
        <tr>
            <td scope="row">HGru4Rec</td>
            <td>hgru4rec.py</td>
            <td>Quadrana et al., Method based on the gru4rec algorithm. To model the interactions of a user within a session, it utilizes RNNs based on a single GRU layer, RecSys 2017.<br>
            </td>
        </tr>
        <tr>
            <td scope="row">IIRNN</td>
            <td>ii_rnn.py</td>
            <td>Ruocco et al., Method extending a session-based recommender built on RNN, called intra-session RNN, by using a second RNN that is called inter-session RNN, DLRS 2017.<br>
            </td>
        </tr>
        <tr>
            <td scope="row">NCSF</td>
            <td>ncfs.py</td>
            <td>Hu et al., Method using three encoders to model inter-session context, intra-session context, and to integrate the information of the intra-session context and the inter-session context for item prediction, IEEE Intelligent Systems, 2018.
            </td>
        </tr>
        <tr>
            <td scope="row">NSAR</td>
            <td>nsar.py</td>
            <td>Phuong et al., Method using RNNs to encode session patterns (short-term user preferences) and user embeddings to represent long-term user preferences across session, IEEE Access 2019.
            </td>
        </tr>
        <tr>
            <td scope="row">SHAN</td>
            <td>shan.py</td>
            <td>Ying et al., Model using a two-layer hierarchical attention network to learn a hybrid representation for each user that combines the long-term and short-term preferences, IJCAI 2018.
            </td>
        </tr>
    </table>
    </div>
    <h3>Factorization-based Methods</h3>
    <div>
    <table class="table table-hover table-bordered">
        <tr>
            <th width="20%" scope="col"> Algorithm</th>
            <th width="12%" class="conf" scope="col">File</th>
            <th width="68%" class="conf" scope="col">Description</th>
        </tr>
        <tr>
            <td scope="row">Factorized Personalized Markov Chains</td>
            <td>fpmc.py</td>
            <td>
                <!--Fpmc combines Markov Chains and traditional user-item matrix factorization in a three dimensional-->
                <!--tensor factorization approach.<br>-->
                Rendle et al., Factorizing Personalized Markov Chains for Next-basket Recommendation. WWW 2010.
            </td>
        </tr>
        <tr>
            <td scope="row">Factored Item Similarity Models</td>
            <td>fism.py</td>
            <td>
                <!--Based on an item-item factorization. It has the advantage of being directly applicable to the-->
                <!--session-based cold-start scenario, where-->
                <!--no explicit user representation can be learned. However, fism does not incorporate sequential-->
                <!--item-to-item transitions like fpmc does. <br>-->
                Kabbur et al., FISM: Factored Item Similarity Models for top-N Recommender Systems, KDD 2013.
            </td>
        </tr>
        <tr>
            <td scope="row">Factorized Sequential Prediction with Item Similarity Models</td>
            <td>fossil.py</td>
            <td>
                <!--Fossil combines fism and factorized Markov chains to incorporate sequential information into the-->
                <!--model.<br>-->
                He and McAuley. Fusing Similarity Models with Markov Chains for Sparse Sequential Recommendation.
                CoRR abs/1609.09152, 2016.
            </td>
        </tr>
        <tr>
            <td scope="row">Session-based Matrix Factorization</td>
            <td>smf.py</td>
            <td>It combines factorized Markov chains with classic matrix factorization. In
                addition, the method considers the
                cold-start situation of session-based recommendation scenarios.
            </td>
        </tr>
    </table>
</div>
</div>
<h2>Related Datasets</h2>
<div>
    Datasets can be downloaded from: <a
        href="https://www.dropbox.com/sh/n281js5mgsvao6s/AADQbYxSFVPCun5DfwtsSxeda?dl=0">https://www.dropbox.com/sh/n281js5mgsvao6s/AADQbYxSFVPCun5DfwtsSxeda?dl=0</a>
    <br>
    <br>
    <table class="tg">
        <tr>
            <td class="tg-0pky">RSC15</td>
            <td class="tg-0pky">The e-commerce dataset used in the 2015 ACM RecSys Challenge.</td>
        </tr>
        <tr>
            <td class="tg-0pky">RETAILROCKET</td>
            <td class="tg-0pky">An e-commerce dataset from the company Retail Rocket.</td>
        </tr>
        <tr>
            <td class="tg-0pky">DIGINETICA</td>
            <td class="tg-0pky">An e-commerce dataset shared by the company Diginetica.</td>
        </tr>
        <tr>
            <td class="tg-0pky">ZALANDO</td>
            <td class="tg-0pky">A private dataset consisting of interaction logs from a European fashion retailer.
            </td>
        </tr>
        <tr>
            <td class="tg-0pky">NOWPLAYING</td>
            <td class="tg-0pky">Music listening logs obtained from Twitter.</td>
        </tr>
        <tr>
            <td class="tg-0pky">30MUSIC</td>
            <td class="tg-0pky">Music listening logs obtained from Last.fm.</td>
        </tr>
        <tr>
            <td class="tg-0pky">AOTM</td>
            <td class="tg-0pky">A public music dataset containing hand-crafted music playlists.</td>
        </tr>
        <tr>
            <td class="tg-0pky">8TRACKS</td>
            <td class="tg-0pky">A private music dataset with hand-crafted playlists.</td>
        </tr>
        <tr>
            <td class="tg-0pky">XING</td>
            <td class="tg-0pky">Interactions of job postings on a career-oriented social networking site, XING, from about three month.</td>
        </tr>
        <tr>
            <td class="tg-0pky">COSMETICS</td>
            <td class="tg-0pky">An e-commerce dataset containing the event history of a cosmetics shop for five months.</td>
        </tr>
        <tr>
            <td class="tg-0pky">LASTFM</td>
            <td class="tg-0pky">A music dataset that contains the entire listening history of almost 1,000 users during five year.</td>
        </tr>
    </table>
</div>
</div>