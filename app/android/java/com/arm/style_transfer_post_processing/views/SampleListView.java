/* Copyright (c) 2020-2021, Arm Limited and Contributors
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.arm.style_transfer_post_processing.views;

import com.google.android.material.tabs.TabLayout;
import androidx.viewpager.widget.ViewPager;
import android.view.View;
import android.widget.AdapterView;

import com.arm.style_transfer_post_processing.FilterDialog;
import com.arm.style_transfer_post_processing.R;
import com.arm.style_transfer_post_processing.SampleLauncherActivity;
import com.arm.style_transfer_post_processing.ViewPagerAdapter;
import com.arm.style_transfer_post_processing.model.Sample;

/**
 * A container for all elements related to the sample view
 */
public class SampleListView {
    private final TabLayout tabLayout;
    public ViewPager viewPager;
    public FilterDialog dialog;

    public SampleListView(SampleLauncherActivity activity) {
        viewPager = activity.findViewById(R.id.viewpager);
        ViewPagerAdapter adapter = new ViewPagerAdapter(activity.getSupportFragmentManager(), activity.samples, new SampleItemClickListener(activity));
        viewPager.setAdapter(adapter);

        dialog = new FilterDialog();
        adapter.setDialog(dialog);

        tabLayout = activity.findViewById(R.id.tabs);
        tabLayout.setupWithViewPager(viewPager);
    }

    /**
     * Show the sample view
     */
    public void show() {
        tabLayout.setVisibility(View.VISIBLE);
        viewPager.setVisibility(View.VISIBLE);
    }

    /**
     * Hide the sample view
     */
    public void hide() {
        tabLayout.setVisibility(View.INVISIBLE);
        viewPager.setVisibility(View.INVISIBLE);
    }
}

/**
 * Click listener for a Sample List Item
 * Start the Native Activity for the clicked Sample
 */
class SampleItemClickListener implements AdapterView.OnItemClickListener {
    private final SampleLauncherActivity activity;

    SampleItemClickListener(SampleLauncherActivity activity) {
        this.activity = activity;
    }

    public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
        String sampleID = ((Sample) parent.getItemAtPosition(position)).getId();
        activity.launchSample(sampleID);
    }
}