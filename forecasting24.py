import bisect
from wst.lib.profile import Profiler
import datetime
import pandas
from wst.lib.html import display_html
from statistics import mean, median, stdev
import scipy.stats as ss
from itertools import combinations
import numpy

def create_out_data_structure(num_entries, num_events):

    out_data= {
        'Wins':[0] * num_entries,
        'ProbWins':[0] * num_entries,
        'Winning Score':[0] * 10000 * num_events,
        'AllScens': [],
    }
    for category in ['Conditionals','FalseConditionals','NumPathConditionals','NumPathFalseConditionals']:
        conds=[]
        for x in range(num_events):
            conds.append([0] * num_entries)
        out_data[category] = conds
    return out_data


def calc_scores(picks_by_player,probs):
    scores=[]
    for picks in picks_by_player:
        so_far=[0]
        for pick,prob in zip(picks,probs):
            if prob == 0:
                so_far = [x + (pick ** 2) for x in so_far]
            elif prob == 1:
                so_far = [x + ((100-pick) ** 2) for x in so_far]
            else:
                so_far = [x + (pick ** 2) for x in so_far] + [x + ((100-pick) ** 2) for x in so_far]
        scores.append( so_far)
    return scores


def calc_probs(probs):
    so_far=[1]
    for prob in probs:
        if prob != 0 and prob != 1:
            so_far = [x * (1-prob) for x in so_far] + [x *(prob) for x in so_far]
    return so_far

def calc_outcomes(probs):
    so_far=[[]]
    for prob in probs:
        if prob == 0:
            so_far = [x+[0] for x in so_far]
        elif prob == 1:
            so_far = [x+[1] for x in so_far]
        else:
            so_far = [x+[0] for x in so_far] + [x+[1] for x in so_far]
    return so_far

def calc_scores_and_probs(picks, probs, breaks):
    print('Calculating Scores and Probs')
    breaks.append(len(picks[0])+1)
    last_break=0
    broken_picks = []
    broken_probs = []
    all_scores = []
    all_probs = []
    all_outcomes = []
    print('Breaking arrays')
    for break_num in breaks:
        broken_picks.append([x[last_break:break_num] for x in picks])
        broken_probs.append(probs[last_break:break_num])
        last_break = break_num
    print('Calculating Broken arrays')
    for idx, pick_subsection in enumerate(broken_picks):
        print(idx, ' Broken array completed')
        all_scores.append( calc_scores(pick_subsection,broken_probs[idx]))
        all_probs.append(calc_probs(broken_probs[idx]))
        all_outcomes.append(calc_outcomes(broken_probs[idx]))

    print('All broken arrays complete')
    all_scores_and_probs = [{'Probs':probs,'Scores':list(map(list, zip(*scores))),'Outcomes':outcomes} for probs,scores,outcomes in zip(all_probs,all_scores,all_outcomes)]
    return all_scores_and_probs

def merge_case(pps, case_prob, case_scores, case_outcome, out_data, output_all_scens=False):

    probs = [x * case_prob for x in pps['Probs']]
    for idx, prob in enumerate(probs):
        outcomes = pps['Outcomes'][idx] + case_outcome
        scores = [x+y for x,y  in zip( pps['Scores'][idx], case_scores )]
        # find the max score and idx
        max_score =min(scores)
        winners = []
        for idx, score in enumerate(scores):
            if score ==max_score:
                winners.append(idx)
        out_data['Winning Score'][max_score] += prob
        prize = 1.0/len(winners)
        for winner in winners:
            if output_all_scens:
                out_data['AllScens'].append([winner, prize, prob, max_score] + outcomes)
            out_data['Wins'][winner]+=prize
            out_data['ProbWins'][winner]+=prize*prob
            for idx,outcome in enumerate(outcomes):
                if outcome == 1:
                    out_data['Conditionals'][idx][winner]+=prize*prob
                    out_data['NumPathConditionals'][idx][winner]+=prize
                else:
                    out_data['FalseConditionals'][idx][winner]+=prize*prob
                    out_data['NumPathFalseConditionals'][idx][winner]+=prize


def count_winners(scores_and_probs1, scores_and_probs2, out_data, required_prob=1, output_all_scens = False):
    print('Presorting')
    sorted_data = sorted(zip(scores_and_probs2['Probs'],scores_and_probs2['Scores'],scores_and_probs2['Outcomes']),reverse=True)
    scores_and_probs2['Outcomes'] = [x[2] for x in sorted_data]
    scores_and_probs2['Scores'] = [x[1] for x in sorted_data]
    scores_and_probs2['Probs'] = [x[0] for x in sorted_data]
#     scores_and_probs1['Scores'] = [numpy.array(x) for x in scores_and_probs1['Scores']]
#     scores_and_probs2['Scores'] = [numpy.array(x) for x in scores_and_probs2['Scores']]
#     scores_and_probs2['Outcomes'] = [x for _,x in sorted(zip(scores_and_probs2['Probs'],scores_and_probs2['Outcomes']),reverse=True)]
#     scores_and_probs2['Scores'] = [x for _,x in sorted(zip(scores_and_probs2['Probs'],scores_and_probs2['Scores']),reverse=True)]
#     scores_and_probs2['Probs'] = sorted( scores_and_probs2['Probs'], reverse = True)
    prob_so_far = 0
    print('Starting main merge case loop')
    time1=datetime.datetime.now()

    for idx, scores2 in enumerate(scores_and_probs2['Scores']):
        print(idx, prob_so_far)
        merge_case(scores_and_probs1, scores_and_probs2['Probs'][idx], scores2, scores_and_probs2['Outcomes'][idx],out_data, output_all_scens)
        prob_so_far += scores_and_probs2['Probs'][idx]
        if prob_so_far > required_prob:
            break
    time2=datetime.datetime.now()
    print('Expensive loop',time2-time1)

def rearrange_events(events, ps, probs):
    print('Sorting Events')
    p_sort = [abs(x-.5)+idx/1000000.0 for idx,x in enumerate(probs)]
    events = [x for _,x in sorted(zip(p_sort,events))]
    probs = [x for _,x in sorted(zip(p_sort,probs))]
    ps = [[x for _,x in sorted(zip(p_sort,entry))] for entry in ps]
    print('Events Sorted')
    return events, ps, probs

def generate_decile_df(winning_scores):
    cum_winning_score=[]
    cum_score = 0
    for x in winning_scores:
        cum_score += x
        cum_winning_score.append( cum_score)
    deciles = []
    for x in [.1,.2,.3,.4,.5,.6,.7,.8,.9]:
        deciles.append([x, bisect.bisect_left(cum_winning_score, x)])
    df = pandas.DataFrame(deciles, columns =['Deciles','Score'])
    return df

if __name__ == '__main__':


    x='''David Seif	Ben Carr	Gary Gambino	Barry Rigal	Kathleen Kollman	Kyle Condron	Tim Lynch	Lois Casaleggi	Alex Silady	Joe Dudman	Jason Waye	Jenny Caplan	Eytan Lenko	Brock Simpson	Brian Schaefer	Scott Musoff	Adam Fass	Bradley Smith	Kathy Richardson	Mike Schramm	Erik Siersdale	Alexandra Epstein	Conor Thompson	Matt Hoy	Casey Remer	Ben Wiles	Bill "Crocodile Avalon" Pennington	Diane Mezzanotte	Dan Ost	Matthew Russell 	Katherine Garcia	Gloria Ambrowiak	Nicole Holliday	Michael McAneny	Andrew Whatlely	Erin Roth	Shrivats Iyer	Kaushik Iyer	Sam Lubchansky	Alex Navissi	Chris Jaunsen	Matt Duchan	Aaron Ellis	Jennifer	Mike Berman	Justin Barleben	Seth Moland-Kovash	Mark Badros	Kristian Schmidt	Nathan Mifsud	Ken Levin	Daniel Michelson-Horowitz 	Adam Broder	Chad Ice	Mark Schiefelbein	S.D. Thompson	Kate Liggett	David Slater	Alex Rose	David Namdar	Matt Balaban	Brian Ecker	Naomi Bloom	Ben Steger	Margaret Friedman	Craig Cepler	Ella Seif	Mia Taylor	David Steinberg	Matthew Hunt	Michael Petkun	Kit Sekelsky	Stan Veuger	William Boyle	Jason Mann	James Bowes	Jim Ellwanger	Aaron Hall	Dimitris Valatsas	Sam Tichnor	Mathew Rideout	Matt Sokol	Katie Bruce	Ash Midalia	Danny Burrows	Pip Butt	Shawn Gardner	Kimberly Fagan	Brad Belsky	Cindy Wiegand	Joel Rosner	Avidan Rose	Gideon Klionsky 	Anthony Label	Michael Lewin	Sarah Barker	Lila Friedland	Maya Seif	Ozzie Zourigui	Matt Sullivan	Andrew Marquis	Xiao-Hu Yan	Eric Distad	Steve Maxon	Travis Hamre	Steve Charnick	Jeff Garst	Alex Guziel	Mel Maisel	Jason Friedlander 	Sam Friedland	Elyssa Friedland	William Friedland	Keith Waites	James Hill III	Arielle and Jason	Rachel and Michael Kay	Weian Wang	Corey Stone	Kathryn Verwillow	Tim Wright	Hillary Seif	Conrad Lumm	Sam Leffell	Jacob Burrows	Rebecca Burrows	John Stryker	Gary Katz	Don Knowles	Raj Dhuwalia	Gerald Larson	Noah Burrows	Benjamin Leffell	Jonathan Huz	Brice Russ	Derek Yi	Candice Day	Murat Tasan
    64	78	85	20	30	70	75	80	90	90	0	85	75	70	75	80	100	100	90	60	50	75	65	90	84	85	85	90	20	65	35	60	90	30	95	60	60	85	90	55	82	74	88	78	33	20	77	80	35	100	90	100	70	91	95	85	80	60	32	100	44	65	65	75	23	75	57	65	88	80	70	75	70	75	80	100	53	50	75	50	100	67	75	79	33	89	65	84	72	55	95	76	85	70	90	82	90	50	85	100	75	40	85	90	100	65	90	85	70	78	75	60	83	90	60	80	80	78	65	95	85	60	90	80	50	30	90	40	75	80	75	100	99	100	71	70	86	50
    18	21	25	20	60	30	60	75	70	20	100	0	5	30	5	20	90	0	50	80	0	65	60	25	72	23	50	45	90	95	60	70	50	44	5	20	30	20	15	3	34	6	90	83	33	85	22	5	5	0	0	0	15	89	75	10	15	40	10	75	2	30	80	10	79	10	41	7	0	30	21	10	0	80	10	80	10	20	70	20	0	13	15	14	33	32	10	76	27	20	66	29	40	10	84	15	40	45	35	50	5	30	20	8	0	35	20	15	20	30	18	35	24	15	5	80	5	5	37	28	22	40	15	20	5	8	90	69	75	65	75	20	50	0	33	10	76	50
    55	52	15	20	20	30	15	40	50	25	100	15	85	100	90	15	80	100	74	20	24	65	30	85	56	87	85	50	15	80	50	56	16	53	30	90	60	80	33	80	53	15	22	23	33	65	89	40	70	75	100	12	50	41	90	75	87	25	15	70	98	75	30	65	80	60	63	91	79	85	56	75	90	95	10	0	31	90	30	50	100	84	50	45	20	75	80	12	65	20	80	23	85	50	8	72	83	69	5	5	65	75	50	85	70	65	85	15	80	68	14	50	67	85	85	80	25	65	50	88	71	50	80	50	65	90	30	34	60	40	60	100	30	85	56	20	24	50
    18	13	33	79	90	50	70	70	95	95	100	90	65	30	80	35	90	100	69	55	80	95	15	95	92	4	0	75	98	65	90	25	90	44	80	90	95	70	75	30	64	87	87	61	66	75	92	70	88	50	20	21	60	90	20	90	90	30	25	30	63	30	70	25	24	65	42	30	14	10	10	10	80	95	20	0	19	90	70	65	100	2	75	14	75	9	20	6	20	20	66	21	20	60	77	32	7	67	15	75	65	65	80	6	10	75	15	10	15	10	68	50	19	75	15	80	40	20	55	14	76	25	35	50	65	90	60	67	60	70	55	100	65	10	22	80	3	50
    36	18	20	85	75	60	60	80	85	85	0	20	82	100	80	85	85	100	30	56	78	75	70	93	72	93	50	82	97	90	90	30	80	4	90	30	95	60	70	92	42	91	90	57	33	65	84	10	61	75	100	76	50	90	25	73	60	70	15	70	98	20	87	35	76	25	37	94	7	75	33	70	90	100	90	100	27	90	95	70	100	38	85	55	80	60	30	74	77	30	70	78	25	40	80	10	92	52	50	60	80	25	95	37	20	85	85	40	90	90	37	50	71	87	75	80	95	33	38	11	90	35	18	30	45	95	20	24	50	70	50	100	50	90	31	30	99	50
    63	84	75	33	45	20	50	50	70	70	0	10	70	70	15	60	20	100	44	30	33	50	10	96	54	6	50	40	50	75	30	20	15	5	15	40	20	30	35	74	68	84	34	43	66	20	35	20	34	50	55	87	70	46	5	20	10	15	80	20	10	35	40	15	34	5	40	36	0	60	90	20	75	50	75	80	47	25	50	50	0	94	90	89	20	43	30	84	20	0	70	34	75	80	10	50	45	76	5	0	60	75	60	28	10	75	15	66	65	23	23	30	35	25	10	20	20	75	67	88	40	10	12	60	65	20	80	0	60	70	35	80	65	85	66	30	34	50
    1	0	5	5	15	10	10	15	15	0	69	1	2	0	0	28	20	0	8	10	36	20	1	15	33	2	0	25	10	5	50	1	10	1	10	15	10	10	10	6	3	11	2	7	22	10	12	10	68	0	0	30	50	39	5	1	5	8	0	20	4	10	10	5	12	0	19	1	0	2	0	0	0	25	30	0	32	1	8	50	0	2	5	0	33	0	0	3	5	40	10	8	1	5	18	10	9	12	55	0	5	30	5	3	0	15	5	5	15	0	14	0	6	5	0	10	10	8	25	9	5	0	0	20	20	8	15	5	10	10	20	0	30	0	6	10	15	0
    63	77	67	32	70	40	20	75	70	85	0	85	10	80	50	20	80	80	33	75	67	30	80	75	75	70	25	75	60	85	60	41	10	55	20	18	80	70	15	65	94	57	36	11	33	40	17	65	88	50	5	69	25	17	2	40	40	55	90	80	14	10	40	25	30	5	55	83	27	25	75	25	95	75	20	80	92	50	65	30	100	83	75	85	33	76	70	25	20	30	50	94	80	73	30	83	6	63	85	0	75	65	70	84	0	35	65	15	90	18	30	0	7	10	67	20	50	10	32	18	25	15	35	50	50	75	90	0	40	30	40	100	30	85	71	80	20	50
    35	56	25	94	60	60	40	75	90	15	100	90	65	55	95	35	10	80	55	70	89	40	25	99	65	63	100	50	65	70	55	60	30	70	70	90	50	50	85	25	75	26	78	49	66	60	69	70	18	25	70	89	38	71	35	50	60	50	0	90	72	30	30	75	66	17	64	39	100	60	67	70	20	50	70	0	93	90	50	30	100	52	50	70	80	56	65	21	30	65	75	87	15	35	14	50	20	46	45	60	95	42	20	75	70	45	25	50	30	0	67	45	60	70	20	20	18	40	58	11	70	20	22	50	40	65	10	15	40	60	40	80	35	80	58	30	50	50
    85	93	90	22	75	65	50	75	10	100	0	99	85	82	90	75	90	100	53	40	50	60	60	75	92	97	100	80	78	99	60	28	80	88	90	23	85	20	90	90	24	68	11	41	88	90	32	80	79	100	85	29	65	38	95	84	45	70	100	100	100	100	30	65	32	95	71	75	88	60	98	25	90	25	95	100	50	75	90	85	100	88	75	94	80	89	85	88	82	70	80	12	90	90	80	50	95	100	95	100	100	78	30	99	80	65	100	85	80	81	99	60	80	86	80	80	95	88	70	60	80	70	65	80	30	45	95	100	75	70	75	100	65	80	85	70	7	50
    3	1	10	0	20	20	10	0	5	0	0	0	0	35	20	25	10	0	3	9	5	15	10	10	10	3	0	35	30	2	45	15	10	2	5	10	5	0	30	5	13	36	1	0	33	15	8	5	16	0	10	9	12	21	2	3	0	10	0	10	2	15	20	10	90	4	20	0	10	10	20	5	0	0	10	0	2	20	10	30	0	2	20	0	33	0	15	2	25	0	5	0	10	4	0	25	0	34	5	0	35	5	25	3	0	15	0	20	5	5	0	0	0	70	2	10	5	8	25	7	10	0	0	20	15	10	5	0	0	15	20	80	30	0	18	10	0	0
    25	15	5	15	40	25	30	45	30	70	0	50	25	27	25	33	10	65	55	70	19	25	5	23	54	7	50	50	35	40	25	45	10	8	20	65	5	20	30	38	39	24	36	37	33	70	28	25	34	0	15	20	25	16	15	18	30	20	20	20	22	15	50	10	36	40	49	5	0	15	20	25	10	20	20	60	12	10	12	60	100	8	30	19	33	33	10	15	20	10	10	12	20	20	10	35	0	50	15	0	35	42	20	15	0	25	5	10	15	30	13	0	7	75	25	20	21	33	32	15	22	40	30	40	35	5	65	50	25	25	20	30	30	30	37	30	22	50
    10	6	0	5	75	5	60	20	10	0	0	5	5	10	10	10	0	0	32	2	14	40	20	60	70	2	0	20	12	1	35	10	12	3	5	32	5	0	15	11	54	9	1	41	22	35	0	10	7	0	0	11	15	6	0	50	25	20	5	10	0	0	10	10	26	52	44	2	0	10	20	0	0	5	10	0	20	50	10	30	0	2	5	10	10	15	0	30	55	0	5	3	15	1	0	22	0	31	5	0	10	5	75	13	0	15	0	25	20	0	0	0	0	13	50	15	15	15	18	15	12	15	25	20	40	5	50	15	0	30	20	90	1	20	28	10	22	0
    22	10	10	5	30	15	10	25	5	65	0	2	25	15	10	25	0	0	52	19	10	20	5	80	28	3	0	20	18	5	15	22	0	6	5	12	5	5	15	4	22	37	2	91	33	80	6	15	30	0	25	17	28	90	5	5	20	20	5	40	1	5	50	15	20	2	39	5	8	10	7	10	0	0	10	0	39	10	10	50	0	17	10	9	75	2	25	6	65	0	10	3	5	5	10	14	9	45	12	0	15	20	20	6	0	35	0	10	10	15	3	30	15	10	5	10	15	22	25	45	29	20	15	20	35	30	5	15	20	40	20	10	10	25	24	20	50	50
    62	79	90	95	25	60	25	75	30	90	100	85	5	70	30	40	90	0	44	85	78	25	40	10	45	96	100	65	67	30	55	18	30	4	80	85	30	50	85	20	40	63	15	83	66	75	94	75	21	75	30	78	78	61	70	33	85	50	75	80	40	40	30	40	15	30	58	60	23	40	85	75	85	75	85	80	79	33	65	30	100	87	80	79	15	81	70	90	25	80	5	34	36	50	90	64	90	43	5	100	60	67	10	40	100	35	90	80	25	91	77	62	73	10	40	80	75	70	25	80	72	18	20	40	45	30	10	100	50	65	35	0	55	85	63	90	90	50
    33	17	0	5	10	0	5	0	3	0	100	10	10	20	70	30	10	0	35	20	30	5	5	70	80	2	0	80	5	5	10	10	8	2	15	72	5	20	20	5	26	4	1	7	33	72	12	20	72	0	5	15	12	41	0	2	0	20	0	20	2	0	15	10	3	20	61	5	0	10	12	35	0	10	5	0	42	25	20	50	0	8	10	4	25	18	10	4	40	0	0	14	10	2	3	15	17	50	5	0	10	12	70	1	0	15	0	30	30	8	4	40	23	10	3	10	15	8	18	11	25	50	16	30	40	20	10	0	0	10	10	10	5	10	32	20	10	50
    26	26	75	17	10	50	20	35	90	10	0	25	10	40	70	50	80	20	32	9	23	15	5	80	43	4	0	50	40	25	25	70	10	8	45	32	60	20	25	21	78	12	41	37	50	50	71	40	12	0	20	41	25	61	15	46	40	20	15	20	33	10	75	20	60	30	38	8	0	20	13	75	10	35	50	80	43	50	75	75	0	18	30	11	75	25	70	11	20	30	20	77	15	40	86	22	10	67	3	50	15	65	60	11	50	75	15	20	35	40	20	50	77	78	24	15	50	33	50	40	60	40	10	30	45	80	50	50	75	60	60	20	40	20	22	20	80	50
    86	91	98	96	15	75	10	40	80	65	0	80	70	80	70	68	90	100	80	80	88	10	95	40	60	95	100	80	40	80	75	80	75	88	30	22	80	80	20	92	63	36	3	17	77	15	65	45	34	75	80	82	78	66	75	51	75	65	50	80	82	80	65	85	42	95	82	90	85	70	75	80	60	60	90	80	40	75	85	60	100	93	70	61	80	76	80	89	75	90	90	86	85	50	17	56	85	87	45	100	90	30	70	7	100	85	100	50	70	92	75	50	66	88	46	80	90	90	68	35	50	70	92	80	35	60	15	25	60	35	60	30	90	75	35	70	15	75
    82	81	65	68	50	40	70	45	60	95	100	90	55	100	80	85	100	100	50	90	70	40	75	80	55	6	50	30	20	95	50	70	20	93	60	78	70	50	95	40	33	78	99	45	88	5	72	60	23	100	100	5	65	99	85	100	80	65	100	80	99	100	40	90	89	90	66	95	83	70	95	25	85	25	80	100	45	50	65	90	100	81	75	84	33	33	95	87	70	90	85	96	75	80	85	28	85	99	85	100	90	37	100	98	60	85	90	85	20	90	99	65	85	90	90	80	73	72	68	55	80	65	80	80	40	75	95	85	60	80	75	100	99	100	78	80	30	50
    30	19	85	75	35	40	70	35	90	75	100	99	40	15	20	70	90	0	80	89	86	25	75	90	56	11	85	90	80	92	25	58	8	60	75	88	20	90	20	28	58	71	63	71	77	30	61	70	34	25	0	75	90	94	5	69	40	20	15	80	98	5	76	25	25	63	49	9	15	80	10	20	50	85	15	20	36	33	80	40	100	7	85	41	20	42	85	62	85	15	25	31	10	45	90	42	15	55	75	70	80	83	100	13	0	15	70	40	40	65	17	55	45	95	20	90	65	55	37	22	33	80	90	40	60	12	25	65	60	90	60	50	70	80	35	20	79	50
    60	65	65	88	65	30	75	5	30	5	69	2	85	29	100	65	20	20	50	4	100	20	60	96	12	98	100	90	76	10	45	68	10	41	85	98	60	30	20	17	47	82	34	63	66	80	24	30	25	100	100	88	50	73	65	45	94	30	83	90	72	50	15	20	75	15	67	70	13	30	70	70	90	65	70	100	17	25	70	75	100	80	50	78	77	91	15	77	30	80	85	13	75	30	10	76	90	56	25	0	70	87	90	84	60	75	80	20	75	34	78	60	80	20	80	90	90	33	55	22	50	75	12	70	95	88	20	75	60	40	75	90	70	20	39	50	30	50
    41	27	25	15	45	55	90	20	80	10	100	80	75	0	15	45	10	20	50	15	64	15	40	75	7	50	0	10	20	64	40	49	8	20	15	50	40	40	25	30	40	22	5	3	22	80	12	50	18	25	5	11	38	65	25	25	40	35	25	0	34	10	50	20	86	60	44	26	18	65	28	75	10	0	40	0	23	40	20	35	0	28	30	20	77	25	5	19	20	20	30	22	30	25	4	44	1	58	95	0	30	50	1	20	10	25	40	25	15	28	4	40	40	15	25	80	23	28	38	50	45	50	14	30	40	60	20	22	20	30	20	80	30	10	25	30	0	50
    20	8	20	15	35	30	50	25	20	5	69	10	20	0	10	45	20	0	50	3	36	10	35	80	5	7	0	65	50	55	10	41	10	60	10	22	30	20	25	10	55	7	50	5	33	30	20	20	2	0	75	32	34	58	5	33	30	10	20	40	50	20	40	50	16	5	39	10	20	10	10	65	0	15	15	0	29	25	20	40	0	9	20	6	20	10	5	44	35	40	15	12	10	30	10	18	12	49	70	20	40	34	5	11	20	35	10	20	30	17	27	40	23	10	2	20	25	15	32	10	60	45	15	20	35	25	5	10	0	15	80	0	40	20	14	30	50	50
    66	65	30	5	30	15	90	0	90	98	100	0	95	20	60	24	10	25	50	12	20	30	25	75	25	86	85	70	45	3	25	28	25	68	10	90	80	20	25	5	73	87	12	11	33	75	23	15	21	0	80	10	30	47	0	50	0	10	90	75	1	85	30	20	87	43	58	85	12	10	65	80	0	0	50	80	10	40	20	25	0	31	35	36	33	40	85	38	35	0	10	88	20	15	80	16	20	60	15	90	10	23	65	23	0	65	5	5	60	5	0	50	70	67	30	10	35	24	45	15	30	60	2	20	30	60	20	80	50	65	60	20	15	15	28	30	60	50
    20	12	5	0	10	40	10	5	50	75	0	90	0	0	10	23	90	0	12	2	10	5	10	10	2	1	50	0	8	5	25	5	5	3	5	65	1	0	80	12	30	56	9	21	33	30	64	30	25	0	0	6	0	31	0	1	10	5	10	20	2	10	2	10	68	12	32	3	7	5	15	1	0	15	40	0	33	25	10	15	0	12	20	13	20	5	5	2	20	0	10	9	15	10	6	86	6	32	35	0	10	32	90	8	30	35	30	15	5	0	19	0	11	10	33	15	50	30	32	15	30	40	18	20	15	10	15	50	20	30	20	30	80	20	23	10	2	50
    28	20	30	0	15	20	2	30	1	5	0	0	20	0	20	35	0	10	55	0	0	2	1	2	0	4	0	25	60	8	5	15	1	3	5	8	5	0	10	8	86	36	0	41	11	0	15	25	5	0	12	11	15	12	15	33	0	25	30	70	4	20	5	20	72	1	43	12	21	20	20	80	10	20	15	0	20	20	30	75	0	30	15	18	66	17	20	21	20	45	40	6	2	5	10	66	13	46	2	0	60	42	10	20	10	15	10	10	40	15	50	40	27	10	1	10	27	40	32	30	5	15	2	20	25	50	80	18	20	35	20	30	40	10	19	20	50	50
    40	59	20	10	50	55	75	35	90	10	100	75	65	37	70	35	10	90	50	5	20	10	50	85	93	94	100	55	85	86	10	30	15	9	10	18	60	60	25	40	78	74	25	91	66	75	68	35	25	25	10	17	35	91	20	80	40	25	60	75	32	50	60	70	65	64	60	45	24	35	52	75	10	85	50	16	13	60	50	90	100	48	35	46	33	71	75	24	70	10	30	18	30	30	90	55	10	48	15	0	60	66	60	70	0	15	25	60	80	9	20	50	15	12	55	10	20	42	45	80	35	45	35	40	20	20	20	15	65	30	70	100	20	80	57	40	25	50
    60	42	20	24	55	60	60	45	80	80	0	80	20	40	75	70	90	85	60	80	70	20	75	30	11	50	50	80	90	65	25	27	20	50	20	13	30	50	15	45	95	46	46	3	66	80	58	60	43	50	75	77	80	5	30	47	30	50	60	80	68	15	80	60	36	42	64	75	23	30	60	35	90	50	75	80	35	80	25	50	100	48	50	40	20	68	75	42	30	80	25	11	50	40	18	86	87	56	50	0	65	77	80	35	0	15	80	60	35	95	73	40	40	89	45	85	70	60	57	20	68	68	20	80	50	60	50	85	50	40	50	70	60	80	59	70	82	50
    28	25	30	10	30	40	10	10	90	80	69	5	78	30	70	45	10	0	22	16	89	55	75	40	85	6	0	90	98	80	30	38	20	77	15	81	25	60	33	25	81	86	79	91	33	85	14	70	34	25	0	31	50	38	15	87	15	25	10	80	11	15	76	30	80	22	56	15	14	70	10	75	15	20	15	0	87	20	70	50	100	11	20	17	10	21	10	35	20	30	15	11	15	20	20	18	19	81	65	100	20	17	75	9	50	65	10	50	2	78	30	50	27	86	10	15	8	60	30	10	60	25	18	20	35	10	5	33	20	80	20	100	30	15	36	0	34	50
    18	11	20	15	25	20	80	10	95	0	69	10	10	0	15	65	90	10	50	20	75	35	15	30	20	4	100	80	40	1	25	31	70	38	5	66	5	0	10	25	51	94	80	81	22	10	32	65	21	0	35	21	30	73	65	9	5	40	0	80	21	15	44	10	66	14	41	5	19	15	14	25	0	50	10	0	77	25	30	50	100	14	20	9	33	12	0	18	70	40	5	4	30	30	14	14	6	23	35	0	10	82	70	2	70	25	5	20	10	6	1	0	14	15	17	10	18	18	25	2	25	25	10	20	45	15	10	50	0	20	10	80	40	20	15	0	23	50'''

    xs = [y.split('\t') for y in x.split('\n')]
    names=xs[0]
    picks =[]
    for idx in range(len(names)):
        picks.append([int(x[idx]) for x in xs[1:]])

    events = [
        'Taiwan','AppleTV','CRBN v XOP','Aus Weather','Delhi AQI','Tax Havens','Millenium Prize','Gaganyaan','WWE','Harden','HRH Harry','Lyft','X Renamed','Govs retire','Bibi',
        'SCOTUS Ret','Auction Record','Mike Johnson','Jets QB','Olympics','Hotels','Mets > Yankees','Nobel','Congress Felon','Zuck Tweet','Surprise Prez','Tesla >= 15 Recalls','Glad>BJuice','Costly cities','F1 CHamp']

    output_deciles              = False
    output_only_live_entries    = False
    output_all_scens            = False or (len(events) < 11)
    output_event_sensitivities  = True
    output_raw_picks            = True
    output_whatifs              = False

    cash_only      = False
    decay          = False
    modify_probs   = True # override averaging method with some common sense manual intervention
    adjust_probs   = False # see what happens if we change aggression
    resolved       = True
    cut_from = 0 # 7, 15, 23
    cut_to = 30
    required_prob = 1.0
    event_nums = list(range(cut_from, cut_to))

    if cash_only:
        entrants = ['Gary Katz','Ben Carr',]

        new_picks = []
        for entrant in entrants:
            new_picks.append(picks[names.index(entrant)])
        names=entrants
        picks=new_picks

    header=''
    picks=[[x[event_num]   for event_num in event_nums] for x in picks]
    events=[events[event_num] for event_num in  event_nums]
    num_entries = len(picks)
    num_events=len(events)
    header += 'Event Cut: ' + ('All events' if len( event_nums) == 30 else str(event_nums))+'<br>'
    header += 'Entry Count: ' + str(num_entries)+'<br>'
    header += 'Event Count: ' + str(num_events)+'<br>'
    header += 'Case Probability Target: ' + ('All Cases' if required_prob ==1.0 else "{0:.0%}".format(required_prob))+'<br>'

    if 0:
        probs_source = 'Mean'
        probs = [mean( [row[event] for row in picks])/100 for event in range(num_events)]
    elif 0:
        probs_source = 'Median'
        probs = [max(1,min(99,median( [row[event] for row in picks])))/100 for event in range(num_events)]
    elif 1:
        probs_source = 'Median Extreme'
        probs = [max(1,min(99,median( [row[event] for row in picks])))/100 for event in range(num_events)]
        probs = [-2*(x**3) + 3*(x**2) for x in probs]
    elif 0:
        probs_source = 'MedianSidePool'
        new_picks17 = []
        for entrant in entrants:
            new_picks17.append(picks[names.index(entrant)])
        probs = [max(1,min(99,median( [row[event] for row in new_picks17])))/100 for event in range(num_events)]

    else:
        probs_source = 'Ben Carr'
#         probs_source = ['Ben Carr','David Seif']
#         probs = [0] * num_events
#         for ps in probs_source:
#         probs = [x/100 for x in picks[names.index(probs_source)]]
        probs = [x/100 for x in picks[names.index(probs_source)]]

    if decay:
        probs_source += ' Decayed'
        start_date = datetime.date(2021,12,31)
        today = datetime.date.today()
#             events = ['Congress Resign','Cavs','Oscars','SpaceIL','Jenner','Harvard','Pulitzer','Snowfall','Bitcoin','Eurovision','Markle','Trump Tweets','Microsoft',
#                       'Pope','Roe v Wade','New Country','Lion King','GDP Growth','Dow 30k','Dead Prez','Peace','Tesla','Maduro & Co','Nobel','NFL Scores','Rugby','Beto','deGrom','Aussie PM','2 Cat 4s',]
        prob_sets = {
#             'Carrie Lam':[datetime.date(2020,3,17), 1],
#             'NEO':[datetime.date(2020,3,31), 0],
        }
        for event, [event_date,towards] in prob_sets.items():
            if event in events:
                if towards == 0:
                    probs[events.index(event)]= probs[events.index(event)] * ( event_date - today)  / (event_date -start_date )
                else:
                    probs[events.index(event)]= 1 - (1 - probs[events.index(event)]) * ( event_date - today)  / (event_date -start_date )
    if modify_probs:
        probs_source += ' Modified'

        prob_sets = {
            'Delhi AQI':.25,
            'Aus Weather':.25,
            'Olympics':.25,
            'Tax Havens':.75

        }
        for event, prob in prob_sets.items():
            if event in events:
                probs[events.index(event)]= prob

    if resolved:
        probs_source += ' Resolved'
        prob_sets = {
            'Taiwan':1.0,
            'AppleTV':0.0,
            'Tax Havens':1.0,

#         'CRBN v XOP','Aus Weather','Delhi AQI','Millenium Prize','Gaganyaan','WWE','Harden','HRH Harry','Lyft','X Renamed','Govs retire','Bibi',
#         'SCOTUS Ret','Auction Record','Mike Johnson','Jets QB','Olympics','Hotels','Mets > Yankees','Nobel','Congress Felon','Zuck Tweet','Surprise Prez','Tesla >= 15 Recalls','Glad>BJuice','Costly cities','F1 CHamp']


        }
        for event, prob in prob_sets.items():
            if event in events:
                probs[events.index(event)]= prob

    header += 'Probability Source: ' + probs_source+'<br>'

    if adjust_probs:
        probs_adj_name= 'Ben Carr'
        prob_adj = 0
        header += 'Probability Adjust: ' + probs_adj_name+' by ' +str(prob_adj) + '<br>'
        for idx, p in enumerate(picks[names.index(probs_adj_name)]):
            picks[names.index('Ben Carr')][idx] = max(0,p-prob_adj) if p < 50 else min(100,p+prob_adj)
    open_events = sum([0 if x == 0.0 or x==1.0 else 1 for x in probs])
    events = ['{0:02.0f}'.format(idx+1)+':'+x for idx,x in enumerate(events)]
    header += '<br>'
    # sort events by likelihood for efficiency
    t1=datetime.datetime.now()
    events, picks, probs = rearrange_events(events, picks, probs)

    if output_whatifs:

        what_if_df = pandas.DataFrame()
        what_if_df['Name'] = names
        total = [0]*len(names)
        for idx1,idx2 in combinations(range(len(probs)),2):
            if probs[idx1] not in [0.0,1.0] or probs[idx2] not in [0.0,1.0] :
                continue
            probs[idx1] = 1 - probs[idx1]
            probs[idx2] = 1 - probs[idx2]

            out_data = create_out_data_structure(num_entries, num_events)
            ps = calc_scores_and_probs(picks, probs, [open_events//2])
            count_winners(ps[0],ps[1],out_data, required_prob,output_all_scens)
            exp_scores  = [sum([pr * ( 100 - pick)*( 100 - pick) + (1-pr) * ( pick)*pick for pr, pick in zip(probs, player_picks) ]) for player_picks in picks]
            ranks = ss.rankdata(exp_scores)
            what_if_df[events[idx1]+events[idx2]] = ranks
            total = [x + y for x,y in zip(total, ranks)]

            probs[idx1] = 1 - probs[idx1]
            probs[idx2] = 1 - probs[idx2]

        what_if_df['Total'] = total
        what_if_df = what_if_df.sort_values(['Total'])
        display_html(header + ' Event Switch ' + what_if_df.to_html())



    # HEAVY LIFTING
    if 1:
        out_data = create_out_data_structure(num_entries, num_events)
        ps = calc_scores_and_probs(picks, probs, [open_events//2])
        #print(ps)
        count_winners(ps[0],ps[1],out_data, required_prob,output_all_scens)
        #print(out_data)
        time2=datetime.datetime.now()
        print(time2-t1)


    exp_scores  = [sum([pr * ( 100 - pick)*( 100 - pick) + (1-pr) * ( pick)*pick for pr, pick in zip(probs, player_picks) ]) for player_picks in picks]
    locked_scores  = [sum([pr * ( 100 - pick)*( 100 - pick) + (1-pr) * ( pick)*pick for pr, pick in zip(probs, player_picks) if pr == 0 or pr == 1.0]) for player_picks in picks]

    out_tbl = [[name,wins,pr_wins, exp,locked] for name, wins, pr_wins,exp,locked in zip(names, out_data['Wins'],out_data['ProbWins'],exp_scores,locked_scores)]
    out_tbl = sorted(out_tbl,key=lambda x: -x[2] if x[2] > 0 else x[-2])
    scoreboard_df = pandas.DataFrame(out_tbl, columns =['Name','Wins', 'PrWins','Expected Scores','Locked Scores'])
    scoreboard_df['Win Prob Rank'] = scoreboard_df['PrWins'].rank(ascending=False)
    scoreboard_df['Expected Score Rank'] = scoreboard_df['Expected Scores'].rank()
    scoreboard_df['Locked Score Rank'] = scoreboard_df['Locked Scores'].rank()
    scoreboard_df=scoreboard_df.rename(columns = {'Wins':'Paths to Victory'})
    display_html('<h1>Win Probs</h1>' + header + scoreboard_df.to_html(index=False,formatters={
    'Paths to Victory': '{:,.1f}'.format,
    'Expected Score Rank':'{:,.0f}'.format,
    'Win Prob Rank':'{:,.0f}'.format,
    'Locked Score Rank':'{:,.0f}'.format,
    'Expected Scores': '{:,.1f}'.format,
    'Locked Scores': '{:,.0f}'.format,
    'PrWins': '{:,.4%}'.format}))

    event_probs_df = pandas.DataFrame([[x,y*100] for x, y in zip(events, probs)],columns = ['Events','Median']).set_index('Events')


    sorted_names = list(scoreboard_df[scoreboard_df['Paths to Victory'] > (0 if output_only_live_entries else -1 )]['Name'].values)

    if output_raw_picks:
        df_pr = pandas.DataFrame(picks, columns =events)
        df_pr['Name'] = names
        df_pr=df_pr.set_index('Name')
        df_pr=df_pr.reindex(sorted_names)
        display_html( '<h1>Raw Picks</h1>' + header + df_pr.to_html())

    if output_event_sensitivities:
        cond_score_ranks = []
        for event_idx in range(len(events)):
            pr = probs[event_idx]
            exp_scores_if_true = [exp_score - pr * ( 100 - pick)*( 100 - pick) - (1-pr) * ( pick)*pick + ( 100 - pick)*( 100 - pick) for exp_score, pick in zip(exp_scores,  [x[event_idx] for x in picks])]
            exp_scores_if_false = [exp_score - pr * ( 100 - pick)*( 100 - pick) - (1-pr) * ( pick)*pick +  pick* pick for exp_score, pick in zip(exp_scores,  [x[event_idx] for x in picks])]
            rank_if_true = ss.rankdata(exp_scores_if_true,method='min')
            rank_if_false = ss.rankdata(exp_scores_if_false,method='min')
            cond_score_ranks.append([str(x)+'_'+str(y) for x,y in zip(rank_if_true, rank_if_false)])
        cond_ranks_df = pandas.DataFrame(cond_score_ranks, columns = names)
        cond_ranks_df.index = events

        sensitivities = {}
        win_if_trues     = [[(0 if x2 == 0 else x1)/(1 if pr_event==0 else pr_event) for x1,x2 in zip(x,out_data['ProbWins'])] for x,pr_event in zip(out_data['Conditionals'],probs)]
        win_if_falses    = [[(0 if x2 == 0 else x1)/(1 if pr_event==1 else (1-pr_event)) for x1,x2 in zip(x,out_data['ProbWins'])] for x,pr_event in zip(out_data['FalseConditionals'],probs)]
        rank_if_trues    = [ss.rankdata(ev,method='min') for ev in [[y *-1 for y in x] for x in win_if_trues ]]
        rank_if_falses   = [ss.rankdata(ev,method='min') for ev in  [[y *-1 for y in x] for x in win_if_falses ]]
        rank_if_strings    = [['{0:.0f}_{1:.0f}'.format(x1,x2)      for x1,x2 in zip(r1,r2)] for r1, r2 in zip(rank_if_trues, rank_if_falses)]
        win_if_diffs     = [[x1-x2                                  for x1,x2 in zip(r1,r2)] for r1, r2 in zip(win_if_trues, win_if_falses)]
        win_if_strings   = [['{0:.3f}-{1:.3f}'.format(x1,x2)        for x1,x2 in zip(r1,r2)] for r1, r2 in zip(win_if_trues, win_if_falses)]
        win_if_diffs_rel = [[0 if (x1+x2 == 0) else (x1-x2)/(x1+x2) for x1,x2 in zip(r1,r2)] for r1, r2 in zip(win_if_trues, win_if_falses)]

    #         t2 = [[(0 if x2 == 0 else x1/x2)/(1 if pr_event==0 else pr_event) for x1,x2 in zip(x,out_data['ProbWins'])] for x,pr_event in zip(out_data['Conditionals'],probs)]

        for df_data, title in [[rank_if_strings,'Rank If'], [win_if_diffs,'Diff in Win Prob'], [win_if_strings,'Conditional Probs'],[win_if_diffs_rel,'Relative Conditional Probs']]:
            df3 = pandas.DataFrame(df_data, columns =names)
            df3['Events'] = events
            df3['Probs'] = probs
            df3['Significance'] = [sum([y*y for y in x]) for x in win_if_diffs]
            df3.index = df3['Events']
#             df3.sort_values(['Significance'],ascending=False,inplace=True)
            df3 = df3[list(df3.columns[-2:]) + sorted_names]
            sensitivities[title] = df3
            display_html('<h1>'+ title + '</h1>' + header + df3.to_html())
        personal_info_html=''
        for picker in sorted_names:
            personal = sensitivities['Relative Conditional Probs'][[picker]].copy()
            personal = personal.join(pandas.DataFrame(df_pr.loc[picker]),rsuffix = 'Prediction')
            personal = personal.join(pandas.DataFrame(event_probs_df),rsuffix = 'Median')
            personal['abs'] = -abs(personal[picker]) + [1e6 if x in [0.0, 100.0] else 0 for x in personal['Median'] ]
            personal = personal.sort_values('abs')
            personal = personal.drop(['abs'],axis = 1)
            personal['Diff']  = personal[picker+'Prediction'] - personal['Median']
            personal = personal.join(sensitivities['Conditional Probs'][[picker]],rsuffix = 'Conditional Probs')
            personal = personal.join(sensitivities['Rank If'][[picker]],rsuffix = 'Win Prob Rank If')
            personal = personal.join(cond_ranks_df[[picker]],rsuffix = 'Score Rank If')
            personal = personal.rename(columns = {picker:picker+'Importance'})
            personal.columns = [x.replace(picker,'') for x in personal.columns]
            personal_info_html +='<h3>'+picker +'</h3>' + personal.to_html()+'<br>'
        display_html('<h1>Personal Tables</h1>' + header  + personal_info_html)
    live_event_num = len(probs) - sum([1 if (x ==0 or x==1) else 0 for x in probs])

    if output_all_scens:
        df5 = pandas.DataFrame(out_data['AllScens'],columns = ['Winner','Winners','Prob','Score']+ events)
        df5['Winner'] = [names[x] for x in df5['Winner']]
        df5.sort_values(['Winner'],inplace=True)
        abridged_cols = df5.columns[0:4 + live_event_num]
#         df5.drop(prob_sets.keys(),axis=1)
#         df5.sort_values(['Winner','Prob'],inplace=True,ascending = [True,False])
        display_html('<h2>All Scenarios</h2>' + header + df5[abridged_cols].to_html())

    if output_deciles:
        df = generate_decile_df(out_data['Winning Score'])
        display_html( '<h1>Expected Winning Score Distribution Deciles</h1>' + header + df.to_html())
