import tensorflow as tf

MAX_PLAYLIST_LENGTH = 5  # cfg.MAX_PLAYLIST_LENGTH


def get_candidate_features():
    """
    candiate tower features
    """

    candidate_features = {
        "track_uri_can": tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "track_name_can": tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "artist_uri_can": tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "artist_name_can": tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "album_uri_can": tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "album_name_can": tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "duration_ms_can": tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "track_pop_can": tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "artist_pop_can": tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "artist_genres_can": tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "artist_followers_can": tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        # new
        # "track_pl_titles_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "track_danceability_can": tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "track_energy_can": tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "track_key_can": tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "track_loudness_can": tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "track_mode_can": tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "track_speechiness_can": tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "track_acousticness_can": tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "track_instrumentalness_can": tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "track_liveness_can": tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "track_valence_can": tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "track_tempo_can": tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "track_time_signature_can": tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    }

    return candidate_features


def get_all_features(MAX_PLAYLIST_LENGTH):
    """
    features for both towers
    """
    feats = {
        # ===================================================
        # candidate track features
        # ===================================================
        "track_uri_can": tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "track_name_can": tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "artist_uri_can": tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "artist_name_can": tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "album_uri_can": tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "album_name_can": tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "duration_ms_can": tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "track_pop_can": tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "artist_pop_can": tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "artist_genres_can": tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "artist_followers_can": tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        # "track_pl_titles_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "track_danceability_can": tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "track_energy_can": tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "track_key_can": tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "track_loudness_can": tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "track_mode_can": tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "track_speechiness_can": tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "track_acousticness_can": tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "track_instrumentalness_can": tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "track_liveness_can": tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "track_valence_can": tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "track_tempo_can": tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "track_time_signature_can": tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        # ===================================================
        # summary playlist features
        # ===================================================
        "name_pl": tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        "collaborative_pl": tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        # 'num_pl_followers_src' : tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "total_duration_ms_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "num_tracks_pl": tf.io.FixedLenFeature(
            dtype=tf.float32, shape=()
        ),  # num_pl_songs_new | n_songs_pl_new
        "num_artists_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        "num_albums_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        # 'avg_track_pop_pl_new' : tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        # 'avg_artist_pop_pl_new' : tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        # 'avg_art_followers_pl_new' : tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
        # ===================================================
        # ragged playlist features
        # ===================================================
        # bytes / string
        "track_uri_pl": tf.io.FixedLenFeature(
            dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)
        ),
        "track_name_pl": tf.io.FixedLenFeature(
            dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)
        ),
        "artist_uri_pl": tf.io.FixedLenFeature(
            dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)
        ),
        "artist_name_pl": tf.io.FixedLenFeature(
            dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)
        ),
        "album_uri_pl": tf.io.FixedLenFeature(
            dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)
        ),
        "album_name_pl": tf.io.FixedLenFeature(
            dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)
        ),
        "artist_genres_pl": tf.io.FixedLenFeature(
            dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)
        ),
        # "tracks_playlist_titles_pl": tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)),
        # Float List
        "duration_ms_pl": tf.io.FixedLenFeature(
            dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)
        ),
        "track_pop_pl": tf.io.FixedLenFeature(
            dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)
        ),
        "artist_pop_pl": tf.io.FixedLenFeature(
            dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)
        ),
        "artist_followers_pl": tf.io.FixedLenFeature(
            dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)
        ),
        "track_danceability_pl": tf.io.FixedLenFeature(
            dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)
        ),
        "track_energy_pl": tf.io.FixedLenFeature(
            dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)
        ),
        "track_key_pl": tf.io.FixedLenFeature(
            dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)
        ),
        "track_loudness_pl": tf.io.FixedLenFeature(
            dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)
        ),
        "track_mode_pl": tf.io.FixedLenFeature(
            dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)
        ),
        "track_speechiness_pl": tf.io.FixedLenFeature(
            dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)
        ),
        "track_acousticness_pl": tf.io.FixedLenFeature(
            dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)
        ),
        "track_instrumentalness_pl": tf.io.FixedLenFeature(
            dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)
        ),
        "track_liveness_pl": tf.io.FixedLenFeature(
            dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)
        ),
        "track_valence_pl": tf.io.FixedLenFeature(
            dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)
        ),
        "track_tempo_pl": tf.io.FixedLenFeature(
            dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)
        ),
        "track_time_signature_pl": tf.io.FixedLenFeature(
            dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)
        ),
    }

    return feats


# tf data parsing functions
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = (
    tf.data.experimental.AutoShardPolicy.DATA
)


# data loading and parsing
def full_parse(data):
    # used for interleave - takes tensors and returns a tf.dataset
    data = tf.data.TFRecordDataset(data)
    return data


def parse_tfrecord(example):
    """
    Reads a serialized example from GCS and converts to tfrecord
    """
    feats = get_all_features(MAX_PLAYLIST_LENGTH)

    # example = tf.io.parse_single_example(
    example = tf.io.parse_example(
        example,
        feats
        # features=feats
    )
    return example


def parse_candidate_tfrecord_fn(example):
    """
    Reads candidate serialized examples from gcs and converts to tfrecord
    """
    candidate_features = get_candidate_features()

    # example = tf.io.parse_single_example(
    example = tf.io.parse_example(example, features=candidate_features)

    return example


# get_candidate_features, get_all_features, full_parse, parse_tfrecord, parse_candidate_tfrecord_fn
