#from databricks.connect import DatabricksSession

from ptb_ltc.config.core import config
#from ptb_ltc.config.env import DB_CLUSTER_ID, DB_HOST, DB_TOKEN

# spark = DatabricksSession.builder.profile(config.general.DB_PROFILE_NAME).getOrCreate()
#spark = DatabricksSession.builder.remote(
#    host=DB_HOST,
#    token=DB_TOKEN,
#    cluster_id=DB_CLUSTER_ID,
#).getOrCreate()