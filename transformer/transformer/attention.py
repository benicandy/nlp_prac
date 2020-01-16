import tensorflow as tf

class MultiheadAttention(tf.keras.models.Model):
    """
    Multi-head Attention のモデル
    model = MultiheadAttention(
        hidden_dim=512,
        head_num=8,
        dropout_rate=0,1,
    )
    model(query, memory, mask, training=True)
    """

    def __init__(self, hidden_dim: int, head_num: int, dropout_rate: float, *args, **kwargs):
        """
        コンストラクタ
        :param hidden_dim: 隠れ層及び出力の次元
            head_num の倍数である必要がある
        :param head_num: ヘッドの数
        :param dropout_rate: ドロップアウトする確率
        """
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.dropout_rate = dropout_rate

        self.q_dense_layer = tf.keras.layers.Dense(hidden_dim, use_bias=False, name='q_dense_layer')
        self.k_dense_layer = tf.keras.layers.Dense(hidden_dim, use_bias=False, name='k_dense_layer')
        self.v_dense_layer = tf.keras.layers.Dense(hidden_dim, use_bias=False, name='v_dense_layer')
        self.output_dense_layer = tf.keras.layers.Dense(hidden_dim, use_bias=False, name='output_dense_layer')
        self.attention_dropout_layer = tf.keras.layers.Dropout(dropout_rate)

    def call(
            self,
            input: tf.Tensor,
            memory: tf.Tensor,
            attention_mask: tf.Tensor,
            training: bool,
    ) -> tf.Tensor:
        """
        モデルの実行を行う
        :param input: query のテンソル
        :param memory: query に情報を与える memory のテンソル
        :param attention_mask: attention weight に適用される mask
            shape = [batch_size, 1, q_length, k_length] のもの
            pad 等無視する部分が True となるようなものを指定する
        :param training: 学習時か推論時かのフラグ
        """
        q = self.q_dense_layer(input)  # [batch_size, q_length, hidden_dim]
        k = self.k_dense_layer(memory)  # [batch_size, m_length, hidden_dim]
        v = self.v_dense_layer(memory)

        # 各行列を head_num 個に分割する
        q = self._split_head(q)  # [batch_size, head_num, q_length, hidden_dim/head_num]
        k = self._split_head(k)  # [batch_size, head_num, m_length, hidden_dim/head_num]
        v = self._split_head(v)  # [batch_size, head_num, m_length, hidden_dim/head_num]

        depth = self.hidden_dim // self.head_num  # 結果が int 型になる
        q *= depth ** -0.5  # for scaled dot production

        # ここで q と k の内積を取ることで，query と key の関連度のようなものを計算する
        logit = tf.matmul(q, k, transpose_b=True)  # [batch_size, head_num, q_length, k_length]
        logit += tf.to_float(attention_mask) * input.dtype.min  # mask は pad 部分などが 1 , 他は 0

        # softmax を取ることで正規化する
        attention_weight = tf.nn.softmax(logit, name='attention_weight')
        attention_weight = self.attention_dropout_layer(attention_weight, training=training)

        # 重みに従って value から情報を引いてくる
        attention_output = tf.matmul(attention_weight, v)  # [batch_size, head_num, q_length, hidden_dim/head_num]
        attention_output = self._combine_head(attention_output)  # [batch_size, q_length, hidden_dim]
        return self.output_dense_layer(attention_output)

    def _split_head(self, x: tf.Tensor) -> tf.Tensor:
        """
        入力の tensor の hidden_dim の次元をいくつかのヘッドに分割する
        入力 shape: [batch_size, length, hidden_dim] の時
        出力 shape: [batch_size, head_num, length, hidden_dim//head_num]
        となる
        """
        with tf.name_scope('split_head'):
            # shape の size を取得
            batch_size, length, hidden_dim = tf.unstack(tf.shape(x))
            x = tf.reshape(x, [batch_size, length, self.head_num, self.hidden_dim // self.head_num])
            return tf.transpose(x, [0, 2, 1, 3])
    
    def _combine_head(self, x: tf.Tensor) -> tf.Tensor:
        """
        入力の tensor の各ヘッドを結合する．_split_head の逆変換

        入力 shape: [batch_size, head_num, length, hidden_dim//head_num] の時
        出力 shape: [batch_size, length, hidden_dim]
        となる
        """
        with tf.name_scope('combine_head'):
            batch_size, _, length, _ = tf.unstack(tf.shape(x))
            x = tf.transpose(x, [0, 2, 1, 3])
            return tf.reshape(x, [batch_size, length, self.hidden_dim])


# Multi-head Attention を継承して　Self-Attention を作る
class SelfAttention(MultiheadAttention):
    def call(  # type: ignore
            self,
            input: tf.Tensor,
            attention_mask: tf.Tensor,
            training: bool,
    ) -> tf.Tensor:
        return super().call(
            input=input,
            memory=input,
            attention_mask=attention_mask,
            training=training,
        )


