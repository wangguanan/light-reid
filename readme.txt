- remove rea
  - add args.use_rea
- add colorjior
  - args.use_colorjitor
- fix res50-ibn bug
  - update ibn resnet model
  - use res50-ibn
- combine available for market and duke
- add njust dataset
  - add njust-spr
- add wildtrack
- combine more dataset
  - run msmt+njust
  - run msmt+njust+duke
  - run msmt+njust+market
  - run msmt+njust+duke+market
- add osnet
- fix bugs
  fix msmt combineall bug
  set osnet training parameters
  - label smooth
  - market 100/ msmt50
  - fix osnet in first 10 epochs
- train osnet with different datasets

---
- copy from 7.1.1.1-bot+osnet
- update vis code
- clean code
- add precision and recall