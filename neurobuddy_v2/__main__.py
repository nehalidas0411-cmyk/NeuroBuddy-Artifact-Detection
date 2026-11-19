"""CLI for single EEG analysis and Team/Athlete comparative workflow."""
from pathlib import Path
import argparse
from .pipeline import process
from .comparative import run_athlete_workflow, run_from_manifest, ComparisonError


def main():
    parser=argparse.ArgumentParser(description='Neuro-buddy V2: single EEG or multi-session Athlete comparative analysis')
    parser.add_argument('input',type=Path,nargs='?',help='Single EEG file or BIDS directory')
    parser.add_argument('--output',type=Path,default=Path('./neurobuddy_reports'))
    parser.add_argument('--player-frames',type=int,default=1500)
    parser.add_argument('--large-threshold-mb',type=int,default=None,
                        help='MiB threshold for streaming. Comparative mode defaults to 256 MiB; use 0 to force consistent streaming.')
    parser.add_argument('--chunk-seconds',type=int,default=60)
    parser.add_argument('--sleep-staging',choices=('if_missing','always','off'),default='if_missing')
    comparison=parser.add_argument_group('Athlete comparative mode')
    comparison.add_argument('--baseline',nargs='+',type=Path,help='At least three distinct baseline EEG .set files')
    comparison.add_argument('--test',type=Path,help='Testing EEG .set file')
    comparison.add_argument('--compare-manifest',type=Path,help='Frontend Team/Athlete JSON manifest')
    comparison.add_argument('--team-manifest',type=Path,help='Frontend batch manifest containing multiple athlete comparisons')
    comparison.add_argument('--team',type=str,help='Team ID for comparative mode')
    comparison.add_argument('--athlete',type=str,help='Athlete ID for comparative mode')
    comparison.add_argument('--z-threshold',type=float,default=3.0,
                            help='Abnormal when absolute z-score is strictly above this value (default 3.0)')
    args=parser.parse_args()
    if args.chunk_seconds<12 or args.player_frames<2 or (args.large_threshold_mb is not None and args.large_threshold_mb<0):
        parser.error('chunk-seconds >=12, player-frames >=2, large-threshold-mb >=0 required')
    mode=bool(args.compare_manifest or args.team_manifest or args.baseline or args.test)
    if mode:
        if args.input:
            parser.error('Do not pass a positional EEG input in comparative mode.')
        if (args.compare_manifest or args.team_manifest) and (args.baseline or args.test):
            parser.error('Use either --compare-manifest or --baseline/--test, not both.')
        if not args.compare_manifest and not args.team_manifest and not (args.baseline and args.test and args.team and args.athlete):
            parser.error('Comparative mode requires --baseline, --test, --team and --athlete, or --compare-manifest.')
        kwargs={'large_threshold_mb':256 if args.large_threshold_mb is None else args.large_threshold_mb,
                'chunk_seconds':args.chunk_seconds,'sleep_staging':args.sleep_staging,
                'player_frames':args.player_frames,'z_threshold':args.z_threshold}
        try:
            if args.team_manifest:
                from .comparative import run_team_manifest
                report=run_team_manifest(args.team_manifest,args.output,**kwargs)
            elif args.compare_manifest:
                report=run_from_manifest(args.compare_manifest,args.output,**kwargs)
            else:
                report=run_athlete_workflow(args.baseline,args.test,args.output,
                                            team_id=args.team,athlete_id=args.athlete,**kwargs)
        except ComparisonError as exc:
            parser.exit(2,f'Comparison not completed: {exc}\n')
        if args.team_manifest:
            print('Team index: '+str(args.output/'team_report.html'))
            print('Transferable Team ZIP: '+str(args.output/'team_report.zip'))
            if report['processing_status'] != 'COMPLETE':
                raise SystemExit(2)
        else:
            print('Athlete report: '+str(args.output/'athlete_report.html'))
            print('Transferable ZIP: '+str(args.output/'athlete_report.zip'))
    else:
        if args.input is None:
            parser.error('Provide a single EEG input, --baseline/--test, or --compare-manifest.')
        reports=process(args.input,args.output,player_frames=args.player_frames,
                        large_threshold_mb=256 if args.large_threshold_mb is None else args.large_threshold_mb,
                        chunk_seconds=args.chunk_seconds,sleep_staging=args.sleep_staging)
        if any(report['processing_status'] in ('FAILED','INCOMPLETE','NEEDS_REVIEW') for report,_ in reports):
            raise SystemExit(2)

if __name__=='__main__':main()
