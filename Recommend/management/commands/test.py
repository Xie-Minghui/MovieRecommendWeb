from django.core.management.base import BaseCommand
import time
from Recommend.LFM_sql import Train as lfm_train
from Recommend.KNN41 import Train as knn_train


class Command(BaseCommand):
    def handle(self, *args, **options):
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        lfm_train()
        knn_train()
        