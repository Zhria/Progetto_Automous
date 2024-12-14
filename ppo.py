""""
This module contains the implementation of the PPO algorithm.
Ci basiamo sullo pseudocodice presente sul sito di OpenAI per la realizzazione del ppo.
https://spinningup.openai.com/en/latest/algorithms/ppo.html#id7
Utilizzando un Actor-Critic Method. 
Ciò suddivide l'implementazione in 8 passi principali:
1. Inizializzazione dell'ambiente con policy parameters theta_0, e l'inizial value function parameters w_0.
2. Ciclare per k iterazioni
3. Raccogliere un set di traiettorie D_k = {τ_i} con una policy pi_k = pi(theta_k)
4. Calcolare i reward-to-go R_t 
5. Calcolare gli advantage estimates A_t basandoci sulla value function V_{w_k}
6. Aggiornare la policy massimizzando la PPO-Clip objective (Gradient ascent con adam) . Non scriverò la formula che è complessa
7. Aggiornare la value function minimizzando la MSE tra V_{w_k} e R_t (Gradient descent con adam)
8. Fine ciclo.

Implementiamo tutti i passi nella funzione learn.
"""
from rete import ReteNeurale
import tensorflow as tf
from keras import Optimizer 
import tensorflow_probability as tfp

class PPO:
    def learn(self,env):
        #Passo 1 --> Inizializzazione dell'ambiente con policy parameters theta_0, e l'inizial value function parameters w_0.
        #Dobbiamo creare una rete neurale per la policy e per la value function. 
        self.nStati=env.observation_space.shape[0]
        self.stepsPerEpisode=2048
        self.episodesPerBatch=8
        self.gamma=0.95
        self.nEpoche=200
        self.epsilon=0.2
        self.nUpdatesPerIteration=10
        self.nAzioni=env.action_space.n
        self.policyNN=ReteNeurale(self.nStati,self.nAzioni) #Actor
        self.valueNN=ReteNeurale(self.nStati,1) #Critic
        self.policy_optimizer=Optimizer.Adam(learning_rate=0.0005)
        self.value_optimizer=Optimizer.Adam(learning_rate=0.0005)
        #passo 2 ciclare per k iterazioni.
        for k in self.nEpoche:
            states, actions, rewards, rewards_to_go, log_probs =self.collect_trajectories()
            V,latest_log_probs=self.evaluate(states,actions)
            advantage=self.calcAdvantages(rewards_to_go,V)

            with tf.GradientTape() as tape:
                _,latest_log_probs=self.evaluate(states,actions)
                surrogated_loss_1, surrogated_loss_2=self.calcSurrogatedLoss(log_probs,latest_log_probs,advantage)
                policy_loss = -tf.reduce_mean(tf.minimum(surrogated_loss_1, surrogated_loss_2))
                value_loss=tf.reduce_mean(tf.square(rewards_to_go-V)) #MSE tra rewards to go e V
            gradientsPolicy = tape.gradient(policy_loss, self.policyNN.trainable_variables)
            self.policy_optimizer.apply_gradients(zip(gradientsPolicy, self.policyNN.trainable_variables))
            gradientsValue = tape.gradient(value_loss, self.valueNN.trainable_variables)
            self.value_optimizer.apply_gradients(zip(gradientsValue, self.valueNN.trainable_variables))                    


    def forward(self, stato):
        azione=self.policyNN(stato)
        value=self.valueNN(stato)
        return azione,value
    
    def collect_trajectories(self):
        #Passo 3 --> Raccogliere un set di traiettorie D_k = {τ_i} con una policy pi_k = pi(theta_k)
        #Dobbiamo raccogliere un set di traiettorie e per fare ciò dobbiamo raccogliere: stati, azioni, rewards, rewards to go, log_prob delle azioni.
        batch={
            'states':[],
            'actions':[],
            'rewards':[],
            'rewards_to_go':[],
            'log_probs':[],
        }
        stato = self.env.reset()
        done = False
        #Abbiamo un fisso di 8 episodi per batch con 2048 steps per episodio
        for i in range(self.episodesPerBatch):
            for j in range(self.stepsPerEpisode):
                batch['states'].append(stato)
                azione,log_prob=self.getAction(stato)
                #azione sarà un int, mentre log_prob sarà il logaritmo della probabilità dell'azione
                batch['actions'].append(azione)
                batch['log_probs'].append(log_prob)
                stato, reward, done, info = self.env.step(azione)
                #info non usata.
                batch['rewards'].append(reward)
                if done:
                    break #Ha raggiunto il termine dell'episodio.
        #Calcoliamo i rewards to go --> PASSO 4
        batch['rewards_to_go']=self.calcRTG(batch['rewards'])
        #return batch states, actions, rewards, rewards to go, log_probs
        return tf.convert_to_tensor(batch['states'],dtype=tf.float32), tf.convert_to_tensor(batch['actions'],dtype=tf.int32),tf.convert_to_tensor(batch['rewards'],dtype=tf.float32),tf.convert_to_tensor(batch['rewards_to_go'],dtype=tf.float32),tf.convert_to_tensor(batch['log_probs'],dtype=tf.float32)
                
    def getAction(self,stato):
        stato=tf.convert_to_tensor(stato,dtype=tf.uint8)
        azione_pred,_=self.forward(stato)
        azione_prob=tf.nn.softmax(azione_pred,axis=-1)
        dist=tfp.distributions.Categorical(probs=azione_prob)
        azionePresa=dist.sample()
        log_prob=dist.log_prob(azionePresa)
        return azionePresa, log_prob

    def calcRTG(self,rewards):
        #Prendo la formula per calcolare i rewards to go e richiede i cumulative rewards e un fattore di sconto.
        rtg=[]
        for episode_reward in reversed(rewards):
            cumulative_reward=0
            for single_reward in reversed(episode_reward):
                cumulative_reward=single_reward+cumulative_reward*self.gamma
                rtg.append(cumulative_reward)
        return tf.convert_to_tensor(rtg[::-1],dtype=tf.float32)

    def calcAdvantages(self, rtg,values):
        advantages=rtg-tf.stop_gradient(values)
        return (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-10)
    
    def calcSurrogatedLoss(self,log_probs_old, log_probs_new, advantages):
        advantages = tf.stop_gradient(advantages)
        policy_ratio = tf.exp(log_probs_old - log_probs_new)
        surrogated_loss_1 = policy_ratio * advantages
        surrogated_loss_2 = tf.clip_by_value(policy_ratio, clip_value_min=1.0-self.epsilon, clip_value_max=1.0+self.epsilon) * advantages
        return surrogated_loss_1, surrogated_loss_2
    
    def evaluate(self, batch_states,batch_actions):
        V= tf.squeeze(self.valueNN(batch_states))
        mean=self.policyNN(batch_states)
        dist=tfp.distributions.Categorical(probs=tf.nn.softmax(mean,axis=-1))
        log_probs=dist.log_prob(batch_actions)
        return V, log_probs

