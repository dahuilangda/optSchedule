"""Jobshop with maintenance tasks using the CP-SAT solver."""

from __future__ import absolute_import
from __future__ import print_function

import collections
from collections import Counter
from ortools.sat.python import cp_model
import datetime
import pandas as pd
import sys
import os
import time


# def getJobsData(df_task, df_machine):
#     '''    
#         return res
#             res = [  # task = (machine_id, processing_time).
#                 [(0, 4),(1, 16),(2, 3),(3,16)],  # Job0
#                 [(0, 5),(1, 8),(2, 7),(3,4)],  # Job1
#                 [(0, 7),(1, 12),(2, 3),(3,5)],  # Job2
#                 [(2, 3),(3,5)]  # Job3]
#     '''
#     res = []
#     n_jobs = int((len(df_task.T)-1)/3)
#     for i in range(n_jobs):
#         m_v = df_machine.method.to_dict()
#         m_l = {v : k for k, v in m_v.items()}
#         # mechine_id = [df_machine[df_machine.method == x].id.to_string(index=False) for x in df_task.iloc[:, i*3+1]]
#         mechine_id = [m_l[x] for x in df_task.iloc[:, i*3+1] if isinstance(x, str)] # 修改为不依赖于自定义machine编号
#         processing_time = [x for x in df_task.iloc[:, i*3+2]]
#         # print(list(zip(mechine_id, processing_time)))
#         r = [(int(i),int(j)) for i, j in zip(mechine_id, processing_time) if j > 0]
#         res.append(r)
#     return res


def getJobsData(df_task, df_machine, startWork_time=8, offWork_time=16):
    '''    
        return res
            res = [  # task = (machine_id, processing_time).
                [(0, 4),(1, 16),(2, 3),(3,16)],  # Job0
                [(0, 5),(1, 8),(2, 7),(3,4)],  # Job1
                [(0, 7),(1, 12),(2, 3),(3,5)],  # Job2
                [(2, 3),(3,5)]  # Job3]
    '''

    res = []
    n_jobs = int((len(df_task.T)-1)/3)
    for i in range(n_jobs):
        m_v = df_machine.method.to_dict()
        m_l = {v : k for k, v in m_v.items()}
        # mechine_id = [df_machine[df_machine.method == x].id.to_string(index=False) for x in df_task.iloc[:, i*3+1]]
        mechine_id = [m_l[x] for x in df_task.iloc[:, i*3+1] if isinstance(x, str)] # 修改为不依赖于自定义machine编号
        mechine = [x for x in df_task.iloc[:, i*3+1] if isinstance(x, str)]
        processing_time = [x for x in df_task.iloc[:, i*3+2]]
        
        for step, t in enumerate(processing_time):
            try:
                t = workHour(t, df_machine, mechine[step], startWork_time, offWork_time)
                processing_time[step] = t
                # print(t)
            except Exception as e:
                # print(e, 'workHour')
                pass

        # print(list(zip(mechine_id, processing_time)))
        r = [(int(i),int(j)) for i, j in zip(mechine_id, processing_time) if j >= 0]
        res.append(r)
    # print(res)
    return res

def machineName2id(df_machine, s):
    '''
        s : 培养基配置，等
        return 编号
    '''
    m_v = df_machine.method.to_dict()
    m_l = {v : k for k, v in m_v.items()}
    return m_l[s]

def machineName2time(df_machine, s):
    '''
        s : 培养基配置，等
        return 该设备可使用的时间长度, 例如：8， 24等
    '''
    t = machineName2id(df_machine, s)
    m_t = df_machine.time.to_dict()
    return m_t[t]

def generateDate(startDate):
    '''
        s = "%Y-%m-%d %H"
        return datetime
    '''
    task_start_time_list = startDate.split(' ')
    d = task_start_time_list[0].split('-')
    res = datetime.datetime(int(d[0]), int(d[1]), int(d[2]), int(task_start_time_list[1]))
    return res

def workHour(t, df_machine, s, startWork_time=8, offWork_time=16):
    '''
    24小时和8小时的缩放尺度不同
    '''
    try:
        t = int(t)
        task_i = machineName2time(df_machine, s)
        # t = int(t*8/task_i)
        if task_i == 24:
            # if 24%(offWork_time - startWork_time) == 0:
            #     t_scale = 24//(offWork_time - startWork_time)
            # else:
            #     t_scale = 24//(offWork_time - startWork_time) + 1
            t_scale = 24/(offWork_time - startWork_time)

            if t%t_scale == 0:
                t_real = t//t_scale
            else:
                t_real = (t//t_scale) + 1
            # t_real = t

            # if t%t_scale == 0:
            #     t_real = (t_scale) * (offWork_time - startWork_time)
            # else:
            #     t_real = (t//t_scale) + t%t_scale
        
        if task_i == 8:
            t_real = t
        return t_real
    except:
        return 0

# def workHour(startDate, t, df_machine, s):
#     startWork_time = 8
#     offWork_time = 17
#     task_i = machineName2time(df_machine, s)

    
    
#     if task_i == 24:
#         task_start_time = startDate.strftime("%Y-%m-%d %H")
#         task_off_time = (startDate + datetime.timedelta(hours=t))
        
#     if task_i == 8: # 当判断为8小时，下班时间自动增加16小时
#         task_start_time = startDate.strftime("%Y-%m-%d %H")
#         task_start_time_list = task_start_time.split(' ')
#         d = task_start_time_list[0].split('-')
#         off_time = datetime.datetime(int(d[0]), int(d[1]), int(d[2]), 17) # 得到下班时间
#         residue_time = off_time - startDate # 项目开始时间到下班时间还剩多少小时
#          # 任务完成时间
#         if residue_time.seconds/3600 - t < 0:
#             t = t + 16*int(t/8)
#             # t = t + 16
#         task_off_time = startDate + datetime.timedelta(hours=t)
        
#         # task_off = task_off_time.strftime("%Y-%m-%d %H")
#         # task_off_list = task_off.split(' ')
#         # d = task_off_list[0].split('-')
#         # off_time = datetime.datetime(int(d[0]), int(d[1]), int(d[2]), 17) # 得到当天的下班时间
#         # residue_off_time = off_time - task_off_time
        
#         # # if residue_off_time.seconds/3600 < 0:
#         # #     print(residue_off_time.seconds/3600)
        
#     return task_off_time, t

def jobshop_with_maintenance(jobs_data, df_task, df_machine):
    l_r = 0
    try:
        df_r = pd.read_csv('restraints.csv', sep=',')
        df_r.mechine.to_string(index=False).strip()
        l_r = len(df_r)
    except:
        pass

    """Solves a jobshop with maintenance on one machine."""
    # Create the model.
    model = cp_model.CpModel()
    
#     m_v = {'培养基':0, '发酵':1, '离心':2, '纯化':3}
    m_v = df_machine.method.to_dict()
    
#    j_v = {'项目1':0, '项目2':1, '项目3':2, '项目4':3}    
    j_v = {}
    j = 0
    for i,c in enumerate(df_task.columns):
        if i % 3 == 1:
            j_v[j] = c
            j += 1


    # machines_count = 1 + max(task[0] for job in jobs_data for task in job)
    # all_machines = range(machines_count)
    all_machines = set([task[0] for job in jobs_data for task in job])

    # Computes horizon dynamically as the sum of all durations.
    horizon = sum(task[1] for job in jobs_data for task in job)
    if l_r > 0:
        horizon += sum([x for x in df_r.end])
        
    # Named tuple to store information about created variables.
    task_type = collections.namedtuple('Task', 'start end interval')
    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple('assigned_task_type',
                                                'start job index duration')

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    if l_r == 0:
        for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                machine = task[0] # machine的id
                duration = task[1]
                suffix = '_%i_%i' % (job_id, task_id)
                start_var = model.NewIntVar(0, horizon, 'start' + suffix)
                end_var = model.NewIntVar(0, horizon, 'end' + suffix)
                interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                    'interval' + suffix)
                all_tasks[job_id, task_id] = task_type(
                    start=start_var, end=end_var, interval=interval_var)
                machine_to_intervals[machine].append(interval_var)
    else:
        m_l = {v : k for k, v in m_v.items()}
        m_id = [m_l[k] for k in df_r.mechine]

        j_l = {v : k for k, v in j_v.items()}
        j_id = [j_l[k] for k in df_r.jobs]
        m_j_list = list(zip(j_id, m_id))
        for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                machine = task[0]
                duration = task[1]
                suffix = '_%i_%i' % (job_id, task_id)
                
                if (job_id, machine) in m_j_list:
                    j, m = job_id, machine
                    start = [x for x in df_r[(df_r.jobs == j_v[j]) & (df_r.mechine == m_v[m])].start][0]
                    end = [x for x in df_r[(df_r.jobs == j_v[j]) & (df_r.mechine == m_v[m])].end][0]
                    duration = end - start
                    start_var = model.NewIntVar(start, horizon, 'start' + suffix)
                    end_var = model.NewIntVar(end, horizon, 'end' + suffix)
                    
                    interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                        'interval' + suffix)
                    all_tasks[job_id, task_id] = task_type(
                        start=start_var, end=end_var, interval=interval_var)
                    machine_to_intervals[m].append(model.NewIntervalVar(start_var, duration, end_var, 'interval' + suffix))
                else:
                    start_var = model.NewIntVar(0, horizon, 'start' + suffix)
                    end_var = model.NewIntVar(0, horizon, 'end' + suffix)
                    interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                        'interval' + suffix)
                    all_tasks[job_id, task_id] = task_type(
                        start=start_var, end=end_var, interval=interval_var)
                    machine_to_intervals[machine].append(interval_var)

    # Add maintenance interval (machine 0 is not available on time {4, 5, 6, 7}).
#     machine_to_intervals[0].append(model.NewIntervalVar(4, 4, 8, 'weekend_0'))
    # machine_to_intervals[3].append(model.NewIntervalVar(4, 4, 8, 'weekend_0'))

    # Create and add disjunctive constraints.
    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])


    # Precedences inside a job.
    overlay_list = []
    for job_id, job in enumerate(jobs_data):
        overlay = [False if x == 'y' else True for x in df_task.iloc[:, job_id*3+3]]
        co = [False if x == 'c' else True for x in df_task.iloc[:, job_id*3+3]]

        # for task_id in range(len(job) - 1):
        for task_id, task in enumerate(job):
            if task_id != len(job)-1:
                if overlay[task_id] == True and co[task_id] == True:
                    model.Add(all_tasks[job_id, task_id + 1].start >= all_tasks[job_id, task_id].end)

                if co[task_id] == False:
                    model.Add(all_tasks[job_id, task_id+1].start == all_tasks[job_id, task_id].end)

                if overlay[task_id] == False:
                    overlay_list.append(task_id)
                    model.Add(all_tasks[job_id, task_id+1].start >= all_tasks[job_id, task_id].start)

                if overlay[task_id-1] == False and overlay[task_id] == True:
                    if len(overlay_list) != 0:
                        # print(overlay_list)
                        model.Add(all_tasks[job_id, task_id+1].start >= all_tasks[job_id, overlay_list[0]].end)
            
            # if overlay[task_id-1] == False and overlay[task_id] == True and co[task_id] == True:
            #     model.Add(all_tasks[job_id, task_id].start >= all_tasks[job_id, overlay_list[0]].end)

        # for task_id, task in enumerate(job):
        #     if task_id != len(job)-1:
        #         if overlay[task_id] == True and co[task_id] == True:
        #             model.Add(all_tasks[job_id, task_id + 1].start >= all_tasks[job_id, task_id].end)

        #         elif overlay[task_id] == True and co[task_id] == False:
        #             overlay_list.append(task_id)                    
        #             model.Add(all_tasks[job_id, task_id + 1].start >= all_tasks[job_id, task_id].start)

        #         elif overlay[task_id] == False and co[task_id] == True:
        #             model.Add(all_tasks[job_id, task_id+1].start == all_tasks[job_id, task_id].end)

        #     if overlay[task_id-1] == False and overlay[task_id] == True:
        #         if len(overlay_list) > 0:
        #             model.Add(all_tasks[job_id, task_id].start >= all_tasks[job_id, overlay_list[0]].end)


    # Makespan objective.
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, [
        all_tasks[job_id, len(job) - 1].end
        for job_id, job in enumerate(jobs_data)
    ])
    model.Minimize(obj_var)

    # Solve model.
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # # Output solution.
    # if status == cp_model.OPTIMAL:
    #     # Create one list of assigned tasks per machine.
    #     assigned_jobs = collections.defaultdict(list)
    #     for job_id, job in enumerate(jobs_data):
    #         for task_id, task in enumerate(job):
    #             machine = task[0]
    #             assigned_jobs[machine].append(
    #                 assigned_task_type(
    #                     start=solver.Value(all_tasks[job_id, task_id].start),
    #                     job=job_id,
    #                     index=task_id,
    #                     duration=task[1]))

    # Output solution.
    if status == cp_model.OPTIMAL:
        if l_r == 0:
            # Create one list of assigned tasks per machine.
            assigned_jobs = collections.defaultdict(list)
            for job_id, job in enumerate(jobs_data):
                for task_id, task in enumerate(job):
                    machine = task[0]
                    assigned_jobs[machine].append(
                        assigned_task_type(
                            start=solver.Value(all_tasks[job_id, task_id].start),
                            job=job_id,
                            index=task_id,
                            duration=task[1]))
                    # assigned_jobs[machine].append(
                    #     assigned_task_type(
                    #         start=solver.Value(all_tasks[job_id, task_id].start),
                    #         job=job_id,
                    #         index=task_id,
                    #         duration=task[1]))
        else:
            m_l = {v : k for k, v in m_v.items()}
            m_id = [m_l[k] for k in df_r.mechine]

            j_l = {v : k for k, v in j_v.items()}
            j_id = [j_l[k] for k in df_r.jobs]
            m_j_list = list(zip(j_id, m_id))

            assigned_jobs = collections.defaultdict(list)
            for job_id, job in enumerate(jobs_data):
                for task_id, task in enumerate(job):
                    machine = task[0]
                    if (job_id, machine) in m_j_list:
                        j, m = job_id, machine
                        start = [x for x in df_r[(df_r.jobs == j_v[j]) & (df_r.mechine == m_v[m])].start][0]
                        end = [x for x in df_r[(df_r.jobs == j_v[j]) & (df_r.mechine == m_v[m])].end][0]
                        duration = end - start
                        assigned_jobs[machine].append(
                            assigned_task_type(
                                start=solver.Value(all_tasks[job_id, task_id].start),
                                job=job_id,
                                index=task_id,
                                duration=duration))
                    else:
                        assigned_jobs[machine].append(
                            assigned_task_type(
                                start=solver.Value(all_tasks[job_id, task_id].start),
                                job=job_id,
                                index=task_id,
                                duration=task[1]))
        # Create per machine output lines.
        output = ''
        for machine in all_machines:
            # Sort by starting time.
            assigned_jobs[machine].sort()
#             sol_line_tasks = 'Machine ' + str(machine) + ': '
            sol_line_tasks = str(m_v[machine]) + ': \t'
#             sol_line_tasks = str(machine) + ': \t'
            sol_line = '\t'

            for assigned_task in assigned_jobs[machine]:
#                 name = 'job_%i_%i' % (assigned_task.job, assigned_task.index)
                name = '%s\t' %(j_v[assigned_task.job])
                # Add spaces to output to align columns.
                sol_line_tasks += '%-10s' % name
                start = assigned_task.start
                duration = assigned_task.duration

                sol_tmp = '[%i,%i]\t' % (start, start + duration)
                # Add spaces to output to align columns.
                sol_line += '%-10s' % sol_tmp

            sol_line += '\n'
            sol_line_tasks += '\n'
            output += sol_line_tasks
            output += sol_line

        # Finally print the solution found.
        # print('Optimal Schedule Length: %ih' % solver.ObjectiveValue())
        # print(output)
    return output, solver.ObjectiveValue()

# def writeFinalSchedule(final):
#     final_list = []
#     for f in final.split('\n'):
#         line_list = []
#         for i in f.split('\t'):
#             line_list.append(i)
#         print(line_list)
#         final_list.append(line_list)
#     df_tmp = pd.DataFrame(final_list)
#     tmp_name = 'result' + str(time.time()) + '.csv'
#     df_tmp.to_csv(tmp_name, index=False, header=None)
#     return df_tmp

def writeFinalSchedule2(final, df_task):
    final_list = []
    df_final = pd.DataFrame()
    for f in final.split('\n'):
        line_list = []
        for i in f.split('\t'):
            line_list.append(i.strip())
        final_list.append(line_list)
    # print(df_task.columns[range(1,len(df_task.columns), 3)]) # Index(['项目1', '项目2'], dtype='object')

    # for step, f_l in enumerate(final_list):
    #     if f_l[0] != '':
    #         f = list(zip(final_list[step], final_list[step+1]))
    #         # print(f[0][0].split(':')[0]) # task 培养基配置
    #         # print(f[1:]) # jobs and time [('项目2', '[50,52]'), ('', '')]
    #         l = []
    #         for j_name in df_task.columns[range(1,len(df_task.columns), 3)]:
    #             # print([x for x in df_task.fillna(0)[j_name].to_list() if x != 0])
    #             # for t in [x for x in df_task.fillna(0)[j_name].to_list() if x != 0]:
    #                 # print(t) # 培养基配置
    #             df_final[j_name] = [x for x in df_task.fillna(0)[j_name].to_list() if x != 0]
                
    #             # print([x for x in f[1:] if x[0] == j_name])
    #                 # df_final[j_name] = [x for x in f[1:]]
    #             t_name = j_name + '-time'
    #             df_final[t_name] = l
    # print(df_final)
    df_final = pd.DataFrame()
    # len_h = len([x for x in df_task.fillna(0)['项目2'].to_list()])
    len_h = len(df_task)
    for j_name in df_task.columns[range(1,len(df_task.columns), 3)]:
        h_1 = [x for x in df_task.fillna(0)[j_name].to_list() if x != 0] # 得到设备列表
        df_final[j_name] = h_1 + ['']*(len_h- len(h_1))
        t_name = j_name + '-time'
        l = []
        l_n = len(l)
        c = Counter(df_final[j_name])
        i = 0
        for t in df_final[j_name]:
            for step, f_l in enumerate(final_list):
                if f_l[0] != '':
                    f = list(zip(final_list[step], final_list[step+1]))
                    if f[0][0].split(':')[0] == t:
                        for x in f[1:]:
                            if x[0] == j_name:
                                if c[t] == 1:
                                    l.append(x[1])
                                else:
                                    l.append([x[1], i])
            i += 1
        # print(l)
        # l_tmp = []
        # for i in l:
        #     if len(i) == 2:
        #         l_tmp.append(i)

        # df_tmp = pd.DataFrame()
        # df_tmp[j_name] = df_final[j_name]
        # tmp_l = []
        # for i, t in enumerate(df_final[j_name]):
        #     if c[t] == 1:
        #         tmp_l.append(1)
        # print(l)
        l_tmp = []
        # print(l)
        n = -1
        for y in l:
            if len(y) == 2:
                if (not y[0] in l_tmp) and (y[1] != n):
                    l_tmp.append(y[0])
                    n = y[1]
            else:
                l_tmp.append(y)
        # print(len_h, len(l_tmp), len_h- len(h_1))
        df_final[t_name] = l_tmp + ['']*(len_h- len(l_tmp))
    # print(df_final)
    # df_tmp = pd.DataFrame(final_list)
    tmp_name = 'tmp/result' + str(time.time()) + '.csv'
    df_final.to_csv(tmp_name, index=False)
    return tmp_name


def schedule(sn, startDate, st, et, task_i, startWork_time=8, offWork_time=16):
    if sn != '':
        if sn <= st:
            final_list = []
            startWork_time = int(startWork_time)
            offWork_time = int(offWork_time)

            start_time = startDate.strftime("%Y-%m-%d %H")
            start_time_list = start_time.split(' ')
            d = start_time_list[0].split('-')
            on_time = datetime.datetime(int(d[0]), int(d[1]), int(d[2]), int(startWork_time)) # 得到上班班时间
            off_time = datetime.datetime(int(d[0]), int(d[1]), int(d[2]), int(offWork_time)) # 得到下班时间
            task_on_time = startDate + datetime.timedelta(hours=st-sn)

            startWork_time = int(startWork_time)
            offWork_time = int(offWork_time)

            if task_i == 24:
                # if 24%(offWork_time - startWork_time) == 0:
                #     t_scale = 24//(offWork_time - startWork_time)
                # else:
                #     t_scale = 24//(offWork_time - startWork_time) + 1
                t_scale = 24/(offWork_time - startWork_time)
                t_real = int(t_scale*(et-st))
                # t_real = 24*(((et-st)*t_scale)//24) + int(((et-st)*t_scale)%24)
                # t_real = t
                # t_real = ((et-st)*t_scale)//24 + (et-st)//t_scale
                # t_real = 3*(t//24) + 2*(t%24)//16 + ((t%24)%16)//8
                # t_real = 24*(t//24) + 2*(t%24)//16 + ((t%24)%16)//8
#                 print(t_real)
                task_off_time = task_on_time + datetime.timedelta(hours=t_real)

            if task_i == 8:
                
                residue_time = int(off_time.strftime("%Y%m%d%H%M%S")) - int(task_on_time.strftime("%Y%m%d%H%M%S")) # 项目开始时间到下班时间还剩多少小时
                if residue_time <= 0: # 如果已经下班
                    # task_on_time = datetime.datetime(int(d[0]), int(d[1]), int(d[2])+1, 8) # 如果项目起始时间已经下班，那么设置为第二天上班
                    task_on_time = on_time + datetime.timedelta(hours=24) # 如果项目起始时间已经下班，那么设置为第二天上班
                
                residue_time = int(task_on_time.strftime("%Y%m%d%H%M%S")) - int(on_time.strftime("%Y%m%d%H%M%S")) # 项目开始到上班时间到上班剩余多少小时
                if residue_time < 0: # 如果还没有上班
                    # t_real = 24*(et - st)//8 + 8*((et - st)%8)
                    task_on_time = datetime.datetime(int(d[0]), int(d[1]), int(d[2]), startWork_time)
                    # task_off_time = task_on_time + datetime.timedelta(hours=t_real)
                # else: # 如果已经上班

                t_real = et - st
                task_off_time = task_on_time + datetime.timedelta(hours=t_real)
                residue_time = int(off_time.strftime("%Y%m%d%H%M%S")) - int(task_off_time.strftime("%Y%m%d%H%M%S"))
                if residue_time < 0: # 如果已下班
                    t_real += 24 - (offWork_time-startWork_time)
                    # t_real = 24*(t_real//(offWork_time-startWork_time)) + (t_real % (offWork_time-startWork_time))
                task_off_time = task_on_time + datetime.timedelta(hours=t_real)
                
                if int(task_off_time.strftime("%H")) > offWork_time:
                    task_off_time = task_off_time + datetime.timedelta(hours=(24-(int(offWork_time)-int(startWork_time))))
                    # print(task_off_time.strftime("%Y-%m-%d-%H"))
                    # tmp_date = task_off_time.strftime("%Y-%m-%d-%H")
                    # tmp_date_list = tmp_date.split('-')
                    # task_off_time = datetime.datetime(int(tmp_date_list[0]), int(tmp_date_list[1]), int(tmp_date_list[2])+1, 8)
                    # print(task_off_time.strftime("%Y-%m-%d-%H"))

            final_list.append([st, task_on_time])
            final_list.append([et, task_off_time])

    return final_list

if __name__ == '__main__':
    filename = sys.argv[1]
    # startDate = '2020-04-07 10'
    # startWork_time = 8
    # offWork_time = 16

    if len(sys.argv[:]) == 2:
        startWork_time = 8
        offWork_time = 16
        startDate = datetime.datetime.now().strftime("%Y-%m-%d %H")

    if len(sys.argv[:]) == 3:
        startWork_time = 8
        offWork_time = 16
        startDate = sys.argv[2]

    if len(sys.argv[:]) == 4:
        startWork_time = int(sys.argv[2])
        offWork_time = int(sys.argv[3])
        startDate = datetime.datetime.now().strftime("%Y-%m-%d %H")

    if len(sys.argv[:]) == 5:
        startWork_time = int(sys.argv[3])
        offWork_time = int(sys.argv[4])
        startDate = sys.argv[2]



    df_task = pd.read_excel(filename, sheet_name='Sheet1')
    df_machine = pd.read_excel(filename, sheet_name='Sheet2')
    res = getJobsData(df_task, df_machine, startWork_time, offWork_time)
    # print(res)
    final, total_time = jobshop_with_maintenance(res, df_task, df_machine)
    # print(final)
    
    # writeFinalSchedule(final)
    tmp_name = writeFinalSchedule2(final, df_task)

    df_1 = pd.read_csv(tmp_name)
    df_m = df_1.iloc[:,range(0, len(df_1.T), 2)]
    df_t = df_1.iloc[:,range(1, len(df_1.T), 2)]

    startDate_1 = startDate.split(' ')[0]
    startDate_2 = startDate.split(' ')[-1]
    startDate_list = startDate_1.split('-') + [startDate_2]
    startDate = datetime.datetime(int(startDate_list[0]),int(startDate_list[1]),int(startDate_list[2]),int(startDate_list[3]))
    startDate_tmp = startDate

    l_8 = []
    l_8_m = []
    l_24 = []
    l_24_m = []
    for i in range(len(df_m.T)):
        m = df_m.iloc[:,i]
        t = df_t.iloc[:,i]
        for j in range(len(m)):
            if isinstance(t[j], str):
    #             print(m[j])
                task_i = machineName2time(df_machine, m[j])
                st = int(t[j].split(',')[0][1:])
                et = int(t[j].split(',')[1][:-1])
                
                if task_i == 8:
                    l_8.append([st,et])
                    l_8_m.append(m[j])
    #             print(task_i)
                if task_i == 24:
                    l_24.append([st,et])
                    l_24_m.append(m[j])
    
    # l_8 = sorted(l_8,key=(lambda x:x[0]))
    # l_24_tmp = sorted(l_24,key=(lambda x:x[0]))


    # 得到时间轴和日期的对应关系
    sn = 0
    dict_time = {}

    # task_i_24_list = []
    # for st, et in l_24_tmp:
    #     task_i_24_list.append(st)
    #     task_i_24_list.append(et)

    for st, et in zip(range(0, int(total_time)+1), range(1, int(total_time)+2)):
        
        # if st in task_i_24_list:
        #     task_i = 24
        # else:
        #     task_i = 8

        final_list = schedule(sn, startDate, st, et, 8, startWork_time, offWork_time)
        if final_list is not None:
            dict_time[final_list[0][0]] = final_list[0][1]
            dict_time[final_list[1][0]] = final_list[1][1]
            startDate = final_list[0][1]
            sn = final_list[0][0]
    
    # print(l_8)
    dict_8 = {}
    for st, et in l_8:
        task_i = 8
        if final_list is not None:
            timeline = '[%s,%s]' %(st, et)
    #         print(timeline)
            dateline = '%s --- %s' %(dict_time[st].strftime("%Y-%m-%d %H"), dict_time[et].strftime("%Y-%m-%d %H"))
    #         print(dateline)
            dict_8[timeline] = dateline
    # print(dict_8)

    # sn = 0
    # dict_8 = {}
    # dict_8_tmp = {}
    # for st, et in l_8:
    #     task_i = 8
    #     final_list = schedule(sn, startDate, st, et, task_i)
    #     if final_list is not None:
    #         timeline = '[%s,%s]' %(final_list[0][0], final_list[1][0])
    # #         print(timeline)
    #         dateline = '%s --- %s' %(final_list[0][1].strftime("%Y-%m-%d %H"), final_list[1][1].strftime("%Y-%m-%d %H"))
    # #         print(dateline)
    #         dict_8[timeline] = dateline
    #         dict_8_tmp[final_list[0][0]] = final_list[0][1]
    #         dict_8_tmp[final_list[1][0]] = final_list[1][1]
    # #         print(final_list)
    #         startDate = final_list[0][1]
    #         sn = final_list[0][0]
    # for st, et in zip(range(0, int(total_time)), range(1, int(total_time))):
    #     final_list = schedule(sn, startDate, st, et, task_i)
    #     if final_list is not None:
    #         timeline = '[%s,%s]' %(final_list[0][0], final_list[1][0])
    # #         print(timeline)
    #         dateline = '%s --- %s' %(final_list[0][1].strftime("%Y-%m-%d %H"), final_list[1][1].strftime("%Y-%m-%d %H"))
    # #         print(dateline)
    #         dict_8[timeline] = dateline
    #         dict_8_tmp[final_list[0][0]] = final_list[0][1]
    #         dict_8_tmp[final_list[1][0]] = final_list[1][1]
    # #         print(final_list)
    #         startDate = final_list[0][1]
    #         sn = final_list[0][0]

    sn = 0
    dict_24 = {}
    for st, et in l_24:
        
        if st in dict_time.keys():
            startDate = dict_time[st]
            sn = st
        else:
            for i in range(1,st):
                if st-i in dict_time.keys():
                    startDate = dict_time[st-i]
                    sn = st -i
                    break
        
    #     task_i = 24
    #     final_list = schedule(sn, startDate, st, et, task_i)
    #     if final_list is not None:
    #         timeline = '[%s,%s]' %(final_list[0][0], final_list[1][0])
    #         print(timeline)
    #         dateline = '%s --- %s' %(final_list[0][1].strftime("%Y-%m-%d %H"), final_list[1][1].strftime("%Y-%m-%d %H"))
    # #         print(dateline)
    #         dict_24[timeline] = dateline
    # #         print(final_list[0][1])

    dict_24 = {}
    for st, et in l_24:
        task_i = 24
        if final_list is not None:
            timeline = '[%s,%s]' %(st, et)
    #         print(timeline)
            dateline = '%s --- %s' %(dict_time[st].strftime("%Y-%m-%d %H"), dict_time[et].strftime("%Y-%m-%d %H"))
    #         print(dateline)
            dict_24[timeline] = dateline

    # df_1 = pd.read_csv('result1588239926.9295206.csv')
    # print(df_1)
    # print(df_1.iloc[:,3])
    for i in range(0, len(df_1)+1, 1):
    # for i in range(0, 1000, 1):
        try:
            for step, t in enumerate(df_1.iloc[i, 1::2]):
                try:
                    if isinstance(df_1.iloc[i, step*2],str):
                        if machineName2time(df_machine, df_1.iloc[i, step*2]) == 8:
                            df_1.iloc[i, step*2+1] = dict_8[t]

                        if machineName2time(df_machine, df_1.iloc[i, step*2]) == 24:
                            df_1.iloc[i, step*2+1] = dict_24[t]
                except Exception as e:
                    print(e, 'aaa')
                    # print(type(df_1.iloc[i, step*2]))
                    pass
        except Exception as e:
            pass
        
        # for step, t in enumerate(df_1.iloc[i, 1::2]):
        #     if isinstance(df_1.iloc[i, step*2],str):
        #         if machineName2time(df_machine, df_1.iloc[i, step*2]) == 8:
        #             df_1.iloc[i, step*2+1] = dict_8[t]

        #         if machineName2time(df_machine, df_1.iloc[i, step*2]) == 24:
        #             df_1.iloc[i, step*2+1] = dict_24[t]
    
    final_schedule_name = 'final_schedule_%s.csv' % datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    df_1.to_csv(final_schedule_name)

    print('Optimal Schedule Length: From %s To %s' %(startDate_tmp, dict_time[total_time]))
    with open(final_schedule_name, 'a') as f:
        f.write(', Optimal Schedule Length: From %s To %s' %(startDate_tmp, dict_time[total_time]))

    # 生成calendar
    # df = pd.read_csv('final_schedule_2020_05_12_13_37_57.csv', index_col=0)
    df = df_1

    startTime_list = []
    endTime_list = []
    title_list = []
    for j in range(0, len(df.T), 2):
        # startTime
        for i in df.iloc[:, j+1]:
            if isinstance(i, str):
                startTime = i.split('---')[0].split(' ')[:-1]
                startTime = startTime[0] + 'T' + startTime[1] + ':00:00'
                startTime_list.append(startTime)
                # print(startTime)

                endTime = i.split('---')[1].split(' ')[1:]
                endTime = endTime[0] + 'T' + endTime[1] + ':00:00'
                endTime_list.append(endTime)
                # print(endTime)

        for i in df.iloc[:, j]:
            if isinstance(i, str):
                if not 'Optimal Schedule Length' in i:
                    title_list.append('%s -> %s' %(df.columns[j], i))
                # print('%s_%s' %(df.columns[j], i))
    
    f_str = ''
    for t, s, e in zip(title_list, startTime_list, endTime_list):
        f_str += '{title:\'%s\', start:\'%s\', end:\'%s\'},\n' %(t, s, e)
    with open('template/calendar_templ_head.html', 'r') as f:
        head_lines = []
        head_lines_tmp = f.readlines()
        for line in head_lines_tmp:
            if 'defaultDate' in line:
                line = 'defaultDate: \'%s\',' % startDate_tmp.strftime("%Y-%m-%d")
            head_lines.append(line)
        head_lines = '\n'.join(head_lines)
    with open('template/calendar_templ_tail.html', 'r') as f:
        tail_lines = f.readlines()
        tail_lines = '\n'.join(tail_lines)

    filename = 'final_calendar_%s.html' % datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    with open(filename, 'w') as f:
        f.write(head_lines+f_str+tail_lines)