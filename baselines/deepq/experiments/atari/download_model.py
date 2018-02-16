from __future__ import absolute_import
import argparse
import progressbar

from baselines.common.azure_utils import Container


def parse_args():
    parser = argparse.ArgumentParser(u"Download a pretrained model from Azure.")
    # Environment
    parser.add_argument(u"--model-dir", type=unicode, default=None,
                        help=u"save model in this directory this directory. ")
    parser.add_argument(u"--account-name", type=unicode, default=u"openaisciszymon",
                        help=u"account name for Azure Blob Storage")
    parser.add_argument(u"--account-key", type=unicode, default=None,
                        help=u"account key for Azure Blob Storage")
    parser.add_argument(u"--container", type=unicode, default=u"dqn-blogpost",
                        help=u"container name and blob name separated by colon serparated by colon")
    parser.add_argument(u"--blob", type=unicode, default=None, help=u"blob with the model")
    return parser.parse_args()


def main():
    args = parse_args()
    c = Container(account_name=args.account_name,
                  account_key=args.account_key,
                  container_name=args.container)

    if args.blob is None:
        print u"Listing available models:"
        print
        for blob in sorted(c.list(prefix=u"model-")):
            print blob
    else:
        print u"Downloading {} to {}...".format(args.blob, args.model_dir)
        bar = None

        def callback(current, total):
            nonlocal bar
            if bar is None:
                bar = progressbar.ProgressBar(max_value=total)
            bar.update(current)

        assert c.exists(args.blob), u"model {} does not exist".format(args.blob)

        assert args.model_dir is not None

        c.get(args.model_dir, args.blob, callback=callback)


if __name__ == u'__main__':
    main()
