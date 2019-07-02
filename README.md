<h1>session-rec</h1>
<h2>Introduction</h2>

<b>session-rec</b> is a Python-based framework for building and evaluating recommender systems (Python 3.5.x). It
implements a suite of state-of-the-art
algorithms and baselines for session-based recommendation.
<br><br>
Parts of the framework and its algorithms are based on code developed and shared by:
<ul>
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
To run session-rec, the following libraries are required:
<ul>
    <li>Anaconda 4.X (Python 3.5+)</li>
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
    <li>Python-telegram-bot</li>
</ul>

<h2>Installation</h2>
<h3>
    Using Anaconda (Windows users)
</h3>
<ol>
    <li>Download and Install Anaconda (https://www.anaconda.com/distribution/)</li>
    <li>Run the following commands:
        <ol>
            <li><code>git clone https://github.com/kyraropmet/session-rec.git</code></li>
            From the main folder run: </br>
            <li><code>conda install --yes --file requirements_conda.txt</code></li>
            <li><code>pip install -r requirements_pip.txt</code></li>      
        </ol>
    </li>
</ol>
<h3>
    Using docker (Linux users with a GPU that supports Cuda9)
</h3>
<ol>
    <li>Download and Install Docker (https://www.docker.com/)</li>
    <li>Run the following commands:
        <ol>
            <li><code>docker pull 042019/session-rec-docker</code></li>
            <li><code>git clone https://github.com/kyraropmet/session-rec.git</code></li>
        </ol>
    </li>
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
            Run a configuration with the following command: </br>
            <code>
                ./dpython run_preprocesing.py conf/preprocess/window/rsc15.yml
            </code>
        </li>
    </ol>
    </li>
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
            <b>Using Anaconda:</b> </br>
            Run the following command from the main folder: </br>
            <code>
                python run_config.py conf/in conf/out
            </code></br>
            If you want to run a specific configuration file, run the following command:</br>
            <code>
                python run_config.py conf/example_next.yml
            </code>
        </li>
        <li>
            <b>Using Docker: </b> </br>
            Run the following command from the main folder: </br>
            <code>
                ./dpython run_config.py conf/in conf/out
            </code></br>
            If you want to run a specific configuration file, run the following command:</br>
            <code>
                ./dpython run_config.py conf/example_next.yml
            </code>
        </li>
        <li>
            Results and run times will be displayed and saved to the results folder as config.
        </li>
    </ol>
    <!--</li>-->
    <!--<h3>-->
        <!--<li>Run experiments using the Python scripts (CHECK SCRIPTS)-->
    <!--</h3>-->
    <!--<ol>-->
        <!--<li>-->
            <!--Open and edit one of the run_test*.py scripts in the run_test folder. The usage of all algorithms is-->
            <!--exemplarily shown in the script.-->
            <!--<ul>-->
                <!--<li>-->
                    <!--runtest.py evaluates predictions for single split in terms of just the next item (HR@X and-->
                    <!--MRR@X)-->
                <!--</li>-->
                <!--<li>-->
                    <!--runtestpr.py evaluates predictions for single split in terms of all remaining items in the-->
                    <!--session (P@X, R@X, and MAP@X)-->
                <!--</li>-->
                <!--<li>-->
                    <!--runtestwindow.py evaluates predictions for window split in terms of the next item (HR@X-->
                    <!--and MRR@X)-->
                <!--</li>-->
            <!--</ul>-->
        <!--</li>-->
        <!--<li>-->
            <!--Run the script with the following command:</br>-->
            <!--<code>-->
                <!--THEANO_FLAGS="device=cuda0,floatX=float32" python run_test*.py-->
            <!--</code>-->
        <!--</li>-->
        <!--<li>-->
            <!--Results and run times will be displayed and saved to the results folder as config.-->
        <!--</li>-->
    <!--</ol>-->
    </li>
</ol>


<h2>How to Configure It</h2>
<b>Start from one of the examples in the conf folder.</b>
<h3>Essential Options</h3>
<div>
    <table class="table table-hover table-bordered">
        <tr>
            <th width="12%" scope="col"> Entry</th>
            <th width="16%" class="conf" scope="col">Example</th>
            <th width="72%" class="conf" scope="col">Description</th>
        </tr>
        <tr>
            <td>type</td>
            <td>single</td>
            <td>Values: single (one single training-test split), window (sliding-window protocol), opt (parameters
                optimization).
            </td>
        </tr>
        <tr>
            <td>evaluation</td>
            <td>evaluation_multiple</td>
            <td>Values: evaluation (evaluation in term of the next item), evaluation_last (evaluation in term of the
                last item of the session), evaluation_multiple (evaluation in terms of the remaining items of the
                sessions).
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
            <td>See the example.yml, example_opt.yml and example_hybrid_opt.yml for a complete list of the
                algorithms and their parameters.<br>
            </td>
        </tr>
    </table>
</div>


<h2>How to extend it</h2>
<ol>
    <li>Make your new algorithm class.</li>
    <li>Write the following functions:</li>
    <ul>
        <li>
            __init__()
        </li>
        <li>
            fit()
        </li>
        <li>
            predict_next()
        </li>
        <li>
            clear()
        </li>
    </ul>
</ol>
Tip: look at the implementation of a baseline (e.g.: ar.py).

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
    </table>
</div>
    <h3>Neural Networks</h3>
    <div>
    <table class="table table-hover table-bordered">
        <tr>
            <th width="20%" scope="col"> Algorithm</th>
            <th width="12%" class="conf" scope="col">File</th>
            <th width="68%" class="conf" scope="col">Description</th>
        </tr>
        <tr>
            <td scope="row">Gru4Rec</td>
            <td>gru4rec.py</td>
            <td>Hidasi et al., Recurrent Neural Networks with Top-k Gains for Session-based Recommendations, CoRR
                abs/1706.03847, 2017.<br>
            </td>
        </tr>
        <tr>
            <td scope="row">STAMP</td>
            <td>STAMP.py</td>
            <td>Liu et al., STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation, KDD
                2018.
            </td>
        </tr>
        <tr>
            <td scope="row">NARM</td>
            <td>narm.py</td>
            <td>Li et al., Neural Attentive Session-based Recommendation, CIKM 2017.
            </td>
        </tr>
        <tr>
            <td scope="row">NextItNet</td>
            <td>nextitrec.py</td>
            <td>Yuan et al., A Simple but Hard-to-Beat Baseline for Session-based Recommendations, CoRR
                abs/1808.05163, 2018.
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
    </table>
</div>
    <h3>Statistics</h3>
<div>
    <table class="tg">
        <tr>
            <th class="tg-0pky" width="25%" scope="col">Dataset</th>
            <th class="tg-0pky" width="15%" scope="col">RSC15-S</th>
            <th class="tg-dvpl" width="15%" scope="col">RSC15</th>
            <th class="tg-dvpl" width="15%" scope="col">TMALL</th>
            <th class="tg-dvpl" width="15%" scope="col">RETAILROCKET</th>
            <th class="tg-dvpl" width="15%" scope="col">ZALANDO</th>
        </tr>
        <tr>
            <td class="tg-0pky">Actions</td>
            <td class="tg-0pky">31,708,461</td>
            <td class="tg-dvpl">5,426,961</td>
            <td class="tg-dvpl">13,418,695</td>
            <td class="tg-dvpl">212,182</td>
            <td class="tg-dvpl">4,536,950</td>
        </tr>
        <tr>
            <td class="tg-0pky">Sessions</td>
            <td class="tg-0pky">7,981,581</td>
            <td class="tg-dvpl">1,375,128</td>
            <td class="tg-dvpl">1,774,729</td>
            <td class="tg-dvpl">59,962</td>
            <td class="tg-dvpl">365,126</td>
        </tr>
        <tr>
            <td class="tg-0pky">Items</td>
            <td class="tg-0pky">37,483</td>
            <td class="tg-dvpl">28,582</td>
            <td class="tg-dvpl">425,348</td>
            <td class="tg-dvpl">31,968</td>
            <td class="tg-dvpl">189,328</td>
        </tr>
        <tr>
            <td class="tg-0pky">Timespan in Days</td>
            <td class="tg-0pky">182</td>
            <td class="tg-dvpl">31</td>
            <td class="tg-dvpl">90</td>
            <td class="tg-dvpl">27</td>
            <td class="tg-dvpl">90</td>
        </tr>
        <tr>
            <td class="tg-0pky">Actions per Session</td>
            <td class="tg-0pky">3.97</td>
            <td class="tg-dvpl">3.95</td>
            <td class="tg-dvpl">7.56</td>
            <td class="tg-dvpl">3.54</td>
            <td class="tg-dvpl">12.43</td>
        </tr>
        <tr>
            <td class="tg-0pky">Unique Items per Session</td>
            <td class="tg-0pky">3.17</td>
            <td class="tg-dvpl">3.17</td>
            <td class="tg-dvpl">5.56</td>
            <td class="tg-dvpl">2.56</td>
            <td class="tg-dvpl">8.39</td>
        </tr>
        <tr>
            <td class="tg-0pky">Actions per Day</td>
            <td class="tg-0pky">174,222.31</td>
            <td class="tg-dvpl">175,063.26</td>
            <td class="tg-dvpl">149,096.61</td>
            <td class="tg-dvpl">7,858.59</td>
            <td class="tg-dvpl">50,410.56</td>
        </tr>
        <tr>
            <td class="tg-0pky">Sessions per Day</td>
            <td class="tg-0pky">43,854.84</td>
            <td class="tg-dvpl">44,358.97</td>
            <td class="tg-dvpl">19,719.22</td>
            <td class="tg-dvpl">2220.84</td>
            <td class="tg-dvpl">4056.96</td>
        </tr>
    </table>

</div>
<div>
    <table class="tg">
        <tr>
            <th class="tg-0pky" width="25%" scope="col">Dataset</th>
            <th class="tg-0pky" width="15%" scope="col">8TRACKS</th>
            <th class="tg-dvpl" width="15%" scope="col">30MUSIC</th>
            <th class="tg-dvpl" width="15%" scope="col">AOTM</th>
            <th class="tg-dvpl" width="15%" scope="col">NOWPLAYING&nbsp</th>
            <th class="tg-dvpl" width="15%" scope="col">CLEF</th>
            <!--<th class="tg-0pky" width="25%" scope="col">Dataset</th>-->
            <!--<th class="tg-0pky" width="15%" scope="col">8TRACKS</th>-->
            <!--<th class="tg-dvpl" width="15%" scope="col">30MUSIC</th>-->
            <!--<th class="tg-dvpl" width="15%" scope="col">AOTM</th>-->
            <!--<th class="tg-dvpl" width="15%" scope="col">NOWPLAYING</th>-->
            <!--<th class="tg-dvpl" width="15%" scope="col">CLEF</th>-->
        </tr>
        <tr>
            <td class="tg-0pky">Actions</td>
            <td class="tg-0pky">1,499,645</td>
            <td class="tg-dvpl">638,933</td>
            <td class="tg-dvpl">306,830</td>
            <td class="tg-dvpl">271,177</td>
            <td class="tg-dvpl">5,540,486</td>
        </tr>
        <tr>
            <td class="tg-0pky">Sessions</td>
            <td class="tg-0pky">132,453</td>
            <td class="tg-dvpl">37,333</td>
            <td class="tg-dvpl">21,888</td>
            <td class="tg-dvpl">27,005</td>
            <td class="tg-dvpl">1,644,442</td>
        </tr>
        <tr>
            <td class="tg-0pky">Items</td>
            <td class="tg-0pky">376,422</td>
            <td class="tg-dvpl">210,633</td>
            <td class="tg-dvpl">91,166</td>
            <td class="tg-dvpl">75,169</td>
            <td class="tg-dvpl">742</td>
        </tr>
        <tr>
            <td class="tg-0pky">Timespan in Days</td>
            <td class="tg-0pky">90</td>
            <td class="tg-dvpl">90</td>
            <td class="tg-dvpl">90</td>
            <td class="tg-dvpl">90</td>
            <td class="tg-dvpl">6</td>
        </tr>
        <tr>
            <td class="tg-0pky">Actions per Session</td>
            <td class="tg-0pky">11.32</td>
            <td class="tg-dvpl">17.11</td>
            <td class="tg-dvpl">14.02</td>
            <td class="tg-dvpl">10.04</td>
            <td class="tg-dvpl">3.37</td>
        </tr>
        <tr>
            <td class="tg-0pky">Unique Items per Session</td>
            <td class="tg-0pky">11.31</td>
            <td class="tg-dvpl">14.47</td>
            <td class="tg-dvpl">14.01</td>
            <td class="tg-dvpl">9.38</td>
            <td class="tg-dvpl">3.17</td>
        </tr>
        <tr>
            <td class="tg-0pky">Actions per Day</td>
            <td class="tg-0pky">16,662.72</td>
            <td class="tg-dvpl">7099,26</td>
            <td class="tg-dvpl">3,409.22</td>
            <td class="tg-dvpl">3,013.08</td>
            <td class="tg-dvpl">923,414</td>
        </tr>
        <tr>
            <td class="tg-0pky">Sessions per Day</td>
            <td class="tg-0pky">1,471.70</td>
            <td class="tg-dvpl">414.81</td>
            <td class="tg-dvpl">243.20</td>
            <td class="tg-dvpl">300.06</td>
            <td class="tg-dvpl">274,074</td>
        </tr>
    </table>
</div>
